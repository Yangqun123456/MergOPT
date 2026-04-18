import os
import gc
import torch
from transformers import AutoModelForCausalLM

# Global variable storing current accumulated average model state
SWA_MODEL_STATE = None


def swa_merge(base_model_path, task_name, task_index, task_count, finetuned_model_prefix="MergeBench/Llama-3.2-3B",
              cache_dir="/root/autodl-tmp/huggingface"):
    """
    Merge models using Simple Weight Averaging (SWA) method - incremental implementation to reduce memory usage

    Principle: Equally average all fine-tuned model weights, using incremental formula:
          avg_new = avg_old * (n-1)/n + new_model * 1/n

    Args:
        base_model_path: Base model path
        task_name: Current task name
        task_index: Current task index
        task_count: Current total number of processed tasks
        scaling_coef: Scaling coefficient (not used, kept for interface consistency)
        finetuned_model_prefix: Fine-tuned model prefix
        cache_dir: Cache directory
        previous_model_state: Compatibility parameter, not used but kept for interface consistency

    Returns:
        model: Merged model
    """
    global SWA_MODEL_STATE

    print(f"Performing SWA continuous merging, current task index: {task_index}, accumulated tasks: {task_count}")

    # Force garbage collection to free memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Determine current model weight
    current_weight = 1.0 / task_count
    previous_weight = 1.0 - current_weight

    print(f"Current model weight: {current_weight:.4f}, accumulated model weight: {previous_weight:.4f}")

    # If it's the first task, directly use current fine-tuned model as accumulated model
    if task_index == 0 or SWA_MODEL_STATE is None:
        # Load fine-tuned model for current task
        ft_model_name = f"{finetuned_model_prefix}_{task_name}"
        print(f"First task: directly using fine-tuned model {ft_model_name} as accumulated model")

        ft_model = AutoModelForCausalLM.from_pretrained(
            ft_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            cache_dir=os.path.join(cache_dir, "transformers")
        )

        # Save current fine-tuned model state as accumulated state
        SWA_MODEL_STATE = ft_model.state_dict()

        # Use current fine-tuned model as merged result
        merged_model = ft_model
    else:
        # Load fine-tuned model for current task
        ft_model_name = f"{finetuned_model_prefix}_{task_name}"
        print(f"Loading current task fine-tuned model: {ft_model_name}")

        ft_model = AutoModelForCausalLM.from_pretrained(
            ft_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            cache_dir=os.path.join(cache_dir, "transformers")
        )

        # Incrementally update accumulated average model
        print(f"Performing incremental SWA update: accumulated model * {previous_weight:.4f} + current model * {current_weight:.4f}")

        # Create base model as container for average result
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            cache_dir=os.path.join(cache_dir, "transformers")
        )
        
        # Check for shape mismatches before performing incremental averaging
        mismatched_keys = []
        for key in SWA_MODEL_STATE:
            if key in ft_model.state_dict():
                if SWA_MODEL_STATE[key].shape != ft_model.state_dict()[key].shape:
                    mismatched_keys.append({
                        'key': key, 
                        'swa_shape': list(SWA_MODEL_STATE[key].shape), 
                        'ft_shape': list(ft_model.state_dict()[key].shape)
                    })
        
        if mismatched_keys:
            print("Warning: Found parameters with mismatched shapes:")
            for info in mismatched_keys:
                print(f"Parameter: {info['key']}, SWA shape: {info['swa_shape']}, Fine-tuned model shape: {info['ft_shape']}")

        # Perform incremental averaging parameter by parameter, skip mismatched parameters
        skipped_keys = []
        updated_keys = []
        
        with torch.no_grad():
            for key in SWA_MODEL_STATE:
                if key in ft_model.state_dict():
                    # Check if shapes match
                    if SWA_MODEL_STATE[key].shape != ft_model.state_dict()[key].shape:
                        print(f"Skipping mismatched parameter: {key}")
                        skipped_keys.append(key)
                        continue
                    
                    try:
                        # Ensure data types match
                        curr_param = ft_model.state_dict()[key].to(SWA_MODEL_STATE[key].dtype)
                        
                        # Perform incremental averaging: prev_avg * (n-1)/n + new_model * 1/n
                        SWA_MODEL_STATE[key] = SWA_MODEL_STATE[key] * previous_weight + curr_param * current_weight
                        updated_keys.append(key)
                    except RuntimeError as e:
                        print(f"Error processing parameter {key}: {e}")
                        skipped_keys.append(key)

        # Report processing results
        print(f"Successfully updated parameters: {len(updated_keys)}, skipped parameters: {len(skipped_keys)}")
        
        # Load state dict in non-strict mode, allowing missing parameters
        base_model.load_state_dict(SWA_MODEL_STATE)    
        
        merged_model = base_model

        # Release fine-tuned model memory
        del ft_model, base_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return merged_model


def save_swa_state_to_disk(directory):
    """Save current SWA state to disk"""
    global SWA_MODEL_STATE
    if SWA_MODEL_STATE is not None:
        # Save model state
        state_path = os.path.join(directory, "swa_model_state.pt")
        torch.save(SWA_MODEL_STATE, state_path)
        print(f"Saved SWA state to: {state_path}")
        return True
    return False


def load_swa_state_from_disk(directory):
    """Load SWA state from disk"""
    global SWA_MODEL_STATE
    state_path = os.path.join(directory, "swa_model_state.pt")

    if os.path.exists(state_path):
        # Load model state
        SWA_MODEL_STATE = torch.load(state_path, map_location="cpu")
        print(f"Loaded SWA state: {state_path}")
        return True

    print(f"SWA state file not found: {state_path}")
    return False