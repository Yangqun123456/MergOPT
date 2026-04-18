import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from optimizers.mergopt import MergOPT
from utils.data_utils import load_local_dataset, prepare_dataset_for_training

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.state.global_step % 10 == 0:
            print(f"\rCurrent loss: {loss.item():.4f}", end="")

        return (loss, outputs) if return_outputs else loss


def finetune_model(args, task_name, base_model_path, output_path):
    start_time = time.time()

    optimizer_type = getattr(args, 'optimizer', 'mergopt')
    print(f"\n--- Starting fine-tuning {task_name} task with {optimizer_type.upper()} optimizer (epochs: {args.epochs}) ---")
    base_optimizer_type = getattr(args, 'base_optimizer', 'adamw')
    print(f"Base optimizer type: {base_optimizer_type.upper()}")

    use_fp16 = torch.cuda.is_available()
    print(f"Model loading config - Half-precision: {use_fp16}")
    cache_dir = args.cache_dir or "/root/autodl-tmp/huggingface"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if use_fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        cache_dir=os.path.join(cache_dir, "transformers")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=os.path.join(cache_dir, "transformers")
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure model is in training mode
    model.train()

    if hasattr(args, 'task_dataset') and args.task_dataset:
        dataset = args.task_dataset
    else:
        print(f"Loading {task_name} task dataset from local...")
        train_dataset = load_local_dataset(task_name, 'train')
        if train_dataset is None:
            raise ValueError(f"Unable to load training data for dataset {task_name}")

        train_dataset = prepare_dataset_for_training(train_dataset, task_name)
        dataset = {"train": train_dataset}

    print(f"Dataset info - Train set size: {len(dataset['train'])}, Column names: {dataset['train'].column_names}")

    def tokenize_function(examples):
        max_length = getattr(args, "max_length", 512)
        if "text" in examples:
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        elif "prompt" in examples and "answer" in examples:
            combined_texts = [f"{p}\n{a}" for p, a in zip(examples["prompt"], examples["answer"])]
            return tokenizer(combined_texts, padding="max_length", truncation=True, max_length=max_length)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset['train'].column_names}")

    remove_columns = dataset["train"].column_names.copy()
    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=remove_columns
    )

    sgd_momentum = getattr(args, 'momentum', 0.9)
    sgd_nesterov = getattr(args, 'nesterov', False)
    effective_epochs = getattr(args, 'epochs', 5)

    training_args = TrainingArguments(
        output_dir=f"./temp/temp_trainer_{task_name}",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,  
        num_train_epochs=effective_epochs,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        push_to_hub=False,
        gradient_checkpointing=True,
        bf16=use_fp16,
        fp16=False,
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if base_optimizer_type == 'sgd':
        base_optimizer_class = torch.optim.SGD
        base_optimizer_kwargs = {
            'lr': 0.1,
            'momentum': sgd_momentum,
            'nesterov': sgd_nesterov,
            'weight_decay': 0.0001
        }
        print(f"Using SGD as base optimizer (momentum={sgd_momentum}, nesterov={sgd_nesterov})")
    else:
        base_optimizer_class = torch.optim.AdamW
        base_optimizer_kwargs = {
            'lr': training_args.learning_rate,
            'weight_decay': 0.001
        }
        print("Using AdamW as base optimizer")

    if optimizer_type == 'mergopt':
        optimizer = MergOPT(
            [p for p in model.parameters() if p.requires_grad],
            base_optimizer_class,
            mu=args.mergopt_mu,
            b=args.mergopt_b,
            k_max=args.mergopt_k_max,
            **base_optimizer_kwargs
        )
        trainer.optimizer = optimizer
        print(f"Configured MergOPT optimizer (mu={args.mergopt_mu}, b={args.mergopt_b}, k_max={args.mergopt_k_max}, using {base_optimizer_type.upper()} as base)")

    elif optimizer_type == 'sgd':
        sgd_lr = 0.1
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=sgd_lr,
            momentum=sgd_momentum,
            weight_decay=0.0001,
            nesterov=sgd_nesterov
        )
        trainer.optimizer = optimizer
        print(f"Configured SGD optimizer (lr={sgd_lr:.6f}, momentum=0.9)")

    else:
        print("Using standard AdamW optimizer")

    print("Starting training...")
    trainer.train()
    finetune_time_seconds = time.time() - start_time

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Fine-tuning completed, model saved to: {output_path}")
    return model, finetune_time_seconds