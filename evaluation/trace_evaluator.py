import torch
import importlib
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_utils import load_local_dataset
from tqdm import tqdm

TASK_EVAL_MAP = {
    'C-STANCE': 'eval_CStance',
    'FOMC': 'eval_FOMC',
    'MeetingBank': 'eval_MeetingBank',
    'ScienceQA': 'eval_ScienceQA',
    'NumGLUE-cm': 'eval_NumGLUE_cm',
    'NumGLUE-ds': 'eval_NumGLUE_ds',
    '20Minuten': 'eval_20Minuten'
}

TASK_GEN_CONFIG = {
    'C-STANCE': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    'FOMC': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    'MeetingBank': {'max_new_tokens': 150, 'temperature': 0.7, 'do_sample': True},
    'ScienceQA': {'max_new_tokens': 100, 'temperature': 0.5, 'do_sample': True},
    'NumGLUE-cm': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    'NumGLUE-ds': {'max_new_tokens': 20, 'temperature': 0.1, 'do_sample': False},
    '20Minuten': {'max_new_tokens': 150, 'temperature': 0.7, 'do_sample': True}
}


def extract_answer(text, prompt):
    if prompt in text:
        answer = text[len(prompt):].strip()
    else:
        answer = text.strip()
    return answer


def extract_first_option(text):
    if not text:
        return ""

    match = re.search(r'([A-D])[.、）\)、\s]', text[:20], re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'答案是\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""


def fallback_evaluate(predicted_sequences, ground_truths, is_multiple_choice=False):
    from evaluation.metrics import caculate_accuracy, caculate_rouge

    if is_multiple_choice:
        return {"accuracy": caculate_accuracy(predicted_sequences, ground_truths)}
    else:
        return {"rouge-l": caculate_rouge(predicted_sequences, ground_truths)}


def evaluate_model_accuracy(model_path, dataset_name, device="auto", result_dir=None):
    import datetime
    import os

    print(f"\nEvaluating model {model_path} performance on {dataset_name} dataset")

    if result_dir:
        save_dir = os.path.join(result_dir, "evaluation_outputs")
    else:
        save_dir = os.path.join("./evaluation_results", "generation_outputs")

    os.makedirs(save_dir, exist_ok=True)

    model_name = os.path.basename(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join(save_dir, f"{model_name}_{dataset_name}_{timestamp}.json")

    test_dataset = load_local_dataset(dataset_name, split='eval')
    if test_dataset is None:
        test_dataset = load_local_dataset(dataset_name, split='test')
        if test_dataset is None:
            return {"error": f"Unable to load eval, test, or validation split for dataset {dataset_name}"}

    if "prompt" not in test_dataset.column_names or "answer" not in test_dataset.column_names:
        return {"error": f"Dataset {dataset_name} does not contain required prompt and answer fields"}

    try:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    input_sequences = []
    prompts = []
    ground_truths = []

    for example in test_dataset:
        if "prompt" in example and "answer" in example:
            prompts.append(example["prompt"])
            ground_truths.append(example["answer"])
            if "input" in example:
                input_sequences.append(example["input"])
            else:
                input_sequences.append(example.get("prompt", ""))

    sample_answers = [test_dataset[i]["answer"] for i in range(min(5, len(test_dataset)))]
    is_multiple_choice = all(len(ans.strip()) <= 3 for ans in sample_answers)

    gen_kwargs = TASK_GEN_CONFIG.get(dataset_name, {})
    if not gen_kwargs:
        if is_multiple_choice:
            gen_kwargs = {
                "max_new_tokens": 20,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": False
            }
        else:
            gen_kwargs = {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }

    predicted_sequences = []
    full_outputs = []

    print(f"Starting to generate answers for {dataset_name}...")

    for prompt in tqdm(prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_outputs.append(output_text)
            answer = extract_answer(output_text, prompt)
            predicted_sequences.append(answer)

        except Exception as e:
            print(f"Error generating response: {e}")
            predicted_sequences.append("")
            full_outputs.append("")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_module_name = TASK_EVAL_MAP.get(dataset_name)
    eval_results = {}

    try:
        if eval_module_name:
            eval_module_path = f"evaluation.eval_modules.{eval_module_name}"
            module = importlib.import_module(eval_module_path)

            if dataset_name == "20Minuten":
                eval_results = module.eval(input_sequences, predicted_sequences, ground_truths)
            else:
                eval_results = module.eval(predicted_sequences, ground_truths)
        else:
            print(f"Warning: Cannot find evaluation module for dataset {dataset_name}, using default evaluation")
            eval_results = fallback_evaluate(predicted_sequences, ground_truths, is_multiple_choice)

    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()

        print("Using fallback evaluation method...")
        eval_results = fallback_evaluate(predicted_sequences, ground_truths, is_multiple_choice)

    save_data = {
        "metadata": {
            "model_path": model_path,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "num_samples": len(prompts),
            "generation_parameters": gen_kwargs,
            "is_multiple_choice": is_multiple_choice
        },
        "results": eval_results,
        "samples": []
    }

    for i in range(len(prompts)):
        sample = {
            "prompt": prompts[i],
            "generated_answer": predicted_sequences[i],
            "reference_answer": ground_truths[i],
            "full_output": full_outputs[i]
        }
        save_data["samples"].append(sample)

    try:
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"Saved generated content and evaluation results to: {save_file}")
    except Exception as e:
        print(f"Error saving evaluation result file: {e}")
    
    print(f"{dataset_name} evaluation results: {json.dumps(eval_results, indent=2)}")
    return eval_results


def evaluate_model_loss(model_path, dataset_name, device="auto", result_dir=None):
    import datetime
    import os
    import numpy as np

    print(f"\nEvaluating model {model_path} loss on {dataset_name} dataset")

    if result_dir:
        save_dir = os.path.join(result_dir, "loss_outputs")
    else:
        save_dir = os.path.join("./evaluation_results", "loss_outputs")
    os.makedirs(save_dir, exist_ok=True)

    model_name = os.path.basename(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join(save_dir, f"{model_name}_{dataset_name}_loss_{timestamp}.json")

    test_dataset = load_local_dataset(dataset_name, split='eval')
    if test_dataset is None:
        test_dataset = load_local_dataset(dataset_name, split='test')
        if test_dataset is None:
            return {"error": f"Unable to load eval, test, or validation split for dataset {dataset_name}", "loss": 10.0}

    try:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        return {"error": f"Failed to load model: {e}", "loss": 10.0}

    losses = []

    for example in tqdm(test_dataset, desc="Calculating loss"):
        try:
            prompt = example["prompt"]
            answer = example["answer"]

            combined_text = prompt + answer
            encodings = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512)

            input_ids = encodings["input_ids"].to(model.device)
            attention_mask = encodings["attention_mask"].to(model.device)

            labels = input_ids.clone()

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                if outputs.loss is not None:
                    loss = outputs.loss.item()
                    losses.append(loss)
                else:
                    print(f"Warning: Loss for sample '{prompt[:30]}...' is None")
                    losses.append(5.0)

        except Exception as e:
            print(f"Error calculating loss: {e}")
            losses.append(5.0)

    avg_loss = float(np.mean(losses)) if losses else 10.0

    save_data = {
        "model_path": model_path,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "num_samples": len(test_dataset),
        "avg_loss": avg_loss,
        "individual_losses": losses[:10]
    }

    try:
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"Saved loss results to: {save_file}")
    except Exception as e:
        print(f"Error saving loss result file: {e}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"{dataset_name} average loss: {avg_loss:.4f}")
    return {"loss": avg_loss}