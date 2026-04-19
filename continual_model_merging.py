import datetime
import gc
import json
import os
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.arguments import parse_args
from evaluation.trace_evaluator import evaluate_model_accuracy
from finetune.finetune_utils import finetune_model
from merging.dare import dare_merge, save_dare_vector_to_disk
from merging.model_utils import get_task_vector_dict
from merging.swa import save_swa_state_to_disk, swa_merge
from merging.task_arithmetic import save_cumulative_vector_to_disk, task_arithmetic_merge
from merging.ties_merge import save_ties_vector_to_disk, ties_merge
from utils.data_utils import get_available_datasets, load_local_dataset
from utils.experiment_utils import (
    create_experiment_dir,
    huggingface_login,
    restore_merge_state,
    save_merge_config,
    setup_hf_cache,
    task_specific_batch_sizes,
    task_specific_epochs,
)

PREVIOUS_MODEL_STATE = {"current_state": None}


def run_continual_model_merging(args):
    setup_hf_cache(args.cache_dir)
    huggingface_login(args.token)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"Using GPU: {gpu_id}")

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        cache_dir=os.path.join(args.cache_dir, "transformers"),
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=os.path.join(args.cache_dir, "transformers"),
    )

    task_names = [
        "C-STANCE",
        "FOMC",
        "MeetingBank",
        "ScienceQA",
        "NumGLUE-cm",
        "NumGLUE-ds",
        "20Minuten",
    ]
    print(f"Will use the following 7 datasets as continual model merging tasks: {task_names}")

    available_datasets = get_available_datasets()
    for task in task_names:
        if task not in available_datasets:
            print(f"Warning: Task '{task}' does not exist in datasets directory, please ensure it is downloaded")

    if args.start_task < 0 or args.start_task > len(task_names):
        raise ValueError(f"Starting task index must be between 0-{len(task_names)}")
    if args.end_task >= len(task_names):
        raise ValueError(f"Ending task index must be between 0-{len(task_names)-1}")

    if args.continue_experiment and args.prev_experiment_dir:
        result_dir = args.prev_experiment_dir
        print(f"\n===== Continuing experiment: {result_dir} =====")
        try:
            with open(os.path.join(result_dir, "experiment_results.json"), "r") as f:
                experiment_results = json.load(f)
        except Exception:
            experiment_results = {}

        task_count = args.start_task
        prev_task_idx = args.start_task - 1
        if prev_task_idx >= 0:
            restore_merge_state(
                args.merge_method,
                result_dir,
                prev_task_idx,
                PREVIOUS_MODEL_STATE,
                args.cache_dir,
            )

        all_finetune_times = []
        for key in experiment_results:
            if key.startswith("task_") and "finetune_time_seconds" in experiment_results[key]:
                all_finetune_times.append(experiment_results[key]["finetune_time_seconds"])
    else:
        result_dir = create_experiment_dir(args.save_path)
        print(f"\n===== Creating new experiment: {result_dir} =====")
        experiment_results = {}
        task_count = 0
        all_finetune_times = []

    finetune_models_dir = os.path.join(result_dir, "finetunedModels")
    os.makedirs(finetune_models_dir, exist_ok=True)

    save_merge_config(
        result_dir=result_dir,
        merge_method=getattr(args, "merge_method", "task_arithmetic"),
        base_model=args.base_model,
        task_names=task_names,
        start_task=args.start_task,
        end_task=args.end_task,
        scaling_coef=args.scaling_coef,
        use_default_scaling=getattr(args, "use_default_scaling", False),
        task_vector_from_base=getattr(args, "task_vector_from_base", True),
        continue_experiment=args.continue_experiment,
        prev_experiment_dir=args.prev_experiment_dir if args.continue_experiment else None,
    )

    if args.start_task == len(task_names):
        print(f"\n{'=' * 50}")
        print("All tasks merged, load final model and evaluate performance on all tasks")
        print(f"{'=' * 50}")

        final_model_path = os.path.join(result_dir, f"model_after_task_{len(task_names)}")
        if not os.path.exists(final_model_path):
            raise ValueError(f"Final merged model not found: {final_model_path}")

        print(f"Loading final merged model: {final_model_path}")
        task_results = {}
        print("\nEvaluating final merged model performance on all tasks:")
        for task_name in task_names:
            print(f"\nEvaluating performance on task {task_name}:")
            task_results[task_name] = evaluate_model_accuracy(
                final_model_path,
                task_name,
                result_dir=result_dir,
            )

        experiment_results["final_evaluation"] = {
            "task_results": task_results,
            "completed_tasks": task_names,
            "evaluation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(os.path.join(result_dir, "experiment_results.json"), "w") as f:
            json.dump(experiment_results, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"Final evaluation results saved to: {result_dir}")
        print(f"{'=' * 50}")
        return

    for i in range(args.start_task, args.end_task + 1):
        task_name = task_names[i]
        print(f"\n{'=' * 50}")
        print(f"Processing task {i + 1}/{args.end_task + 1}: {task_name}")
        print(f"{'=' * 50}")

        task_count += 1
        args.epochs = task_specific_epochs.get(task_name, args.epochs)
        args.batch_size = task_specific_batch_sizes.get(task_name, args.batch_size)
        print(f"Setting epochs: {args.epochs}, batch_size: {args.batch_size} for task {task_name}")

        finetune_model_path = os.path.join(finetune_models_dir, f"finetuned_{task_name}")
        finetune_time_seconds = 0

        if os.path.exists(finetune_model_path) and args.continue_experiment:
            print(f"Found existing fine-tuned model: {finetune_model_path}, skipping fine-tuning step")
        else:
            start_model_path = args.base_model
            print(f"Using base model for fine-tuning task {task_name}")

            print(f"Loading dataset {task_name} for fine-tuning...")
            train_dataset = load_local_dataset(task_name, split="train")
            if train_dataset:
                args.task_dataset = {"train": train_dataset}
                _, finetune_time_seconds = finetune_model(
                    args,
                    task_name,
                    start_model_path,
                    finetune_model_path,
                )
                print(f"Fine-tuning task {task_name} completed, time taken {finetune_time_seconds:.2f} seconds")
            else:
                raise ValueError(f"Unable to load training data for dataset {task_name}")
            all_finetune_times.append(finetune_time_seconds)

        print("\nComputing current task vector")
        if args.merge_method == "swa":
            print("Using SWA merge method, skipping task vector computation")
            current_vector_dict = None
        else:
            ft_model = AutoModelForCausalLM.from_pretrained(
                finetune_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                cache_dir=os.path.join(args.cache_dir, "transformers"),
            )

            if args.task_vector_from_base:
                print("Computing task vector: Fine-tuned model - Original pre-trained model")
                current_vector_dict = get_task_vector_dict(ft_model, base_model)
            else:
                print("Computing task vector: Fine-tuned model - Current merged model")
                if i == 0 or not PREVIOUS_MODEL_STATE.get("current_state"):
                    reference_model = base_model
                else:
                    prev_model_path = os.path.join(result_dir, f"model_after_task_{i}")
                    if os.path.exists(prev_model_path):
                        reference_model = AutoModelForCausalLM.from_pretrained(
                            prev_model_path,
                            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                            cache_dir=os.path.join(args.cache_dir, "transformers"),
                        )
                    else:
                        print("Warning: Previous task merged model not found, using base model as reference")
                        reference_model = base_model
                current_vector_dict = get_task_vector_dict(ft_model, reference_model)

            del ft_model
            if "reference_model" in locals() and reference_model is not base_model:
                del reference_model
            gc.collect()

        current_model_path = os.path.join(result_dir, f"model_after_task_{i + 1}")
        scaling_coef_to_use = None if getattr(args, "use_default_scaling", False) else args.scaling_coef

        print(f"Executing merge operation using method: {args.merge_method}...")
        if args.merge_method == "task_arithmetic":
            merged_model = task_arithmetic_merge(
                base_model_path=args.base_model,
                current_vector_dict=current_vector_dict,
                task_index=i,
                task_count=task_count,
                scaling_coef=scaling_coef_to_use,
                cache_dir=args.cache_dir,
            )
            save_cumulative_vector_to_disk(result_dir)
        elif args.merge_method == "ties_merge":
            merged_model = ties_merge(
                base_model_path=args.base_model,
                current_vector_dict=current_vector_dict,
                task_index=i,
                task_count=task_count,
                scaling_coef=scaling_coef_to_use,
                cache_dir=args.cache_dir,
            )
            save_ties_vector_to_disk(result_dir)
        elif args.merge_method == "dare":
            merged_model = dare_merge(
                base_model_path=args.base_model,
                current_vector_dict=current_vector_dict,
                task_index=i,
                task_count=task_count,
                scaling_coef=scaling_coef_to_use,
                drop_rate=0.9,
                mask_strategy="random",
                cache_dir=args.cache_dir,
            )
            save_dare_vector_to_disk(result_dir)
        elif args.merge_method == "swa":
            merged_model = swa_merge(
                base_model_path=args.base_model,
                task_name=task_name,
                task_index=i,
                task_count=task_count,
                finetuned_model_prefix=os.path.join(finetune_models_dir, "finetuned"),
                cache_dir=args.cache_dir,
            )
            save_swa_state_to_disk(result_dir)
        else:
            raise ValueError(f"Unsupported merge method: {args.merge_method}")

        print(f"Saving merged model to: {current_model_path}")
        if torch.cuda.is_available():
            print("Converting model to bfloat16 and saving...")
            merged_model = merged_model.to(torch.bfloat16)
        else:
            print("Converting model to float16 and saving...")
            merged_model = merged_model.to(torch.float16)

        merged_model.save_pretrained(
            current_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
        base_tokenizer.save_pretrained(current_model_path)

        task_results = {}
        eval_model_path = current_model_path

        if not args.evaluate_at_end or i == args.end_task:
            print("\nEvaluating merged model performance on all previous tasks:")
            for prev_task_idx in range(0, i + 1):
                prev_task = task_names[prev_task_idx]
                print(f"\nEvaluating performance on task {prev_task}:")
                task_results[prev_task] = evaluate_model_accuracy(
                    eval_model_path,
                    prev_task,
                    result_dir=result_dir,
                )

            experiment_results[f"task_{i + 1}_{task_name}"] = {
                "task_results": task_results,
                "completed_tasks": task_names[: i + 1],
                "finetune_time_seconds": finetune_time_seconds,
            }

            with open(os.path.join(result_dir, "experiment_results.json"), "w") as f:
                json.dump(experiment_results, f, indent=2)

            if all_finetune_times:
                experiment_results["all_finetune_times_seconds"] = all_finetune_times
                experiment_results["average_finetune_time_seconds"] = sum(all_finetune_times) / len(all_finetune_times)
                with open(os.path.join(result_dir, "experiment_results.json"), "w") as f:
                    json.dump(experiment_results, f, indent=2)
                print(f"\nAll task fine-tune times: {all_finetune_times}")
                print(f"Average fine-tune time: {experiment_results['average_finetune_time_seconds']:.2f} seconds")

        del merged_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base_model, base_tokenizer
    gc.collect()

    print(f"\n{'=' * 50}")
    print(f"Experiment results saved to: {result_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_continual_model_merging(args)
    except Exception as e:
        print(f"\nError during continual model merging: {e}")
        traceback.print_exc()