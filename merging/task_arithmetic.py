import os
import torch
import gc
from transformers import AutoModelForCausalLM
from merging.model_utils import apply_task_vector_dict

# Global variable to store accumulated task vectors
CUMULATIVE_TASK_VECTOR = None


def task_arithmetic_merge(base_model_path, current_vector_dict,
                          task_index=None, task_count=1, scaling_coef=0.2,
                          cache_dir="/root/autodl-tmp/huggingface"):
    """
    Merge models using Task Arithmetic method - execute entirely on CPU to avoid GPU memory shortage

    Principle: θ_merged = θ_base + scaling_coef * τ_cum
    where τ_cum is the cumulative sum of all task vectors

    Args:
        base_model_path: Base model path
        current_vector_dict: Pre-computed current task vector
        task_index: Current task index
        task_count: Total number of processed tasks
        scaling_coef: Task vector scaling coefficient, controls overall merging strength
        cache_dir: Cache directory

    Returns:
        model: Merged model
    """
    global CUMULATIVE_TASK_VECTOR

    print(f"Performing Task Arithmetic continuous merging, current task index: {task_index}, accumulated tasks: {task_count}")

    if scaling_coef is None:
        scaling_coef = 0.2
        print(f"Using Task Arithmetic default scaling coefficient: {scaling_coef}")
        
    # Force garbage collection to free memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load base model to CPU
    print(f"Loading base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        cache_dir=os.path.join(cache_dir, "transformers")
    )

    # Decide how to handle based on task index
    if task_index == 0 or CUMULATIVE_TASK_VECTOR is None:
        # First task, directly use current task vector as accumulated vector
        print("First task or accumulated vector not initialized, setting current task vector as accumulated vector")
        CUMULATIVE_TASK_VECTOR = {k: v.clone()
                                  for k, v in current_vector_dict.items()}
    else:
        # Not the first task, accumulate current task vector to existing accumulated vector
        print("Accumulating current task vector to accumulated vector")

        # Ensure all keys are handled
        all_keys = set(CUMULATIVE_TASK_VECTOR.keys()).union(
            current_vector_dict.keys())

        for key in all_keys:
            if key in CUMULATIVE_TASK_VECTOR and key in current_vector_dict:
                # Both vectors have this key, accumulate
                if CUMULATIVE_TASK_VECTOR[key].dtype != current_vector_dict[key].dtype:
                    current_vector_dict[key] = current_vector_dict[key].to(
                        CUMULATIVE_TASK_VECTOR[key].dtype)

                CUMULATIVE_TASK_VECTOR[key] += current_vector_dict[key]
            elif key in current_vector_dict:
                # Only current vector has this key
                CUMULATIVE_TASK_VECTOR[key] = current_vector_dict[key].clone()

    # Create a copy of accumulated vector to avoid accidental modification of global variable
    combined_vector_dict = {k: v.clone()
                            for k, v in CUMULATIVE_TASK_VECTOR.items()}

    # Release unnecessary variables to save memory
    del current_vector_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Apply merged vector to base model and create final model
    # scaling_coef controls the strength of the entire merging process
    print(f"Applying accumulated task vector to base model (scaling_coef={scaling_coef})...")
    merged_model = apply_task_vector_dict(
        combined_vector_dict, base_model, scaling_coef=scaling_coef)

    # Release merged vector
    del combined_vector_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return merged_model


def save_cumulative_vector_to_disk(directory):
    """Save current accumulated task vector to disk"""
    global CUMULATIVE_TASK_VECTOR
    if CUMULATIVE_TASK_VECTOR is not None:
        vector_path = os.path.join(directory, "cumulative_task_vector.pt")
        torch.save(CUMULATIVE_TASK_VECTOR, vector_path)
        print(f"Saved accumulated task vector to: {vector_path}")
        return vector_path
    return None


def load_cumulative_vector_from_disk(directory):
    """Load accumulated task vector from disk"""
    global CUMULATIVE_TASK_VECTOR
    vector_path = os.path.join(directory, "cumulative_task_vector.pt")
    if os.path.exists(vector_path):
        CUMULATIVE_TASK_VECTOR = torch.load(vector_path)
        print(f"Loaded accumulated task vector: {vector_path}")
        return True
    print(f"Accumulated task vector file not found: {vector_path}")
    return False