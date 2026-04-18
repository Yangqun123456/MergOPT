import os
import torch
import gc
from transformers import AutoModelForCausalLM
from merging.model_utils import apply_task_vector_dict

# Global variable to store accumulated task vectors
DARE_TASK_VECTOR = None

def apply_mask_and_rescale(tensor, drop_rate=0.9, mask_strategy="random"):
    """
    Apply mask and rescale tensor
    
    Args:
        tensor: Input tensor
        drop_rate: Drop ratio (0.0-1.0)
        mask_strategy: Mask strategy ("random" or "magnitude")
        
    Returns:
        Processed tensor
    """
    if drop_rate <= 0 or drop_rate >= 1:
        return tensor.clone()
        
    # Choose mask strategy
    if mask_strategy == "random":
        # Random mask
        mask = torch.rand_like(tensor, dtype=torch.float32) > drop_rate
    elif mask_strategy == "magnitude":
        # Mask based on absolute value magnitude
        abs_tensor = torch.abs(tensor)
        threshold = torch.quantile(abs_tensor.flatten(), drop_rate)
        mask = abs_tensor > threshold
    else:
        raise ValueError(f"Unsupported mask strategy: {mask_strategy}")
    
    # Apply mask
    masked_tensor = tensor.clone() * mask
    
    # Rescale
    keep_ratio = mask.float().mean()
    if keep_ratio > 0:
        scale_factor = 1.0 / keep_ratio
        masked_tensor = masked_tensor * scale_factor
        
    return masked_tensor

def dare_merge(base_model_path, current_vector_dict,
              task_index=None, task_count=1, scaling_coef=0.2,
              cache_dir="/root/autodl-tmp/huggingface", 
              drop_rate=0.9, mask_strategy="random"):
    """
    Merge models using DARE (Drop And REscale) method - incremental implementation to reduce memory usage
    
    Principle: 1. Randomly or based on importance drop some parameters
               2. Rescale remaining parameters to maintain overall impact
               3. Accumulate multiple task vectors
    
    Args:
        base_model_path: Base model path
        current_vector_dict: Pre-computed current task vector
        task_index: Current task index
        task_count: Total number of processed tasks
        scaling_coef: Task vector scaling coefficient
        cache_dir: Cache directory
        previous_model_state: Compatibility parameter, not used but kept for interface consistency
        drop_rate: Proportion of parameters to drop, default 0.9 (90%)
        mask_strategy: Mask generation strategy, options "random" or "magnitude"
        
    Returns:
        model: Merged model
    """
    global DARE_TASK_VECTOR
    
    print(f"Performing DARE continuous merging, current task index: {task_index}, accumulated tasks: {task_count}")
    print(f"Parameters: drop_rate={drop_rate}, mask_strategy={mask_strategy}")
    
    if scaling_coef is None:
        scaling_coef = 0.2
        print(f"Using DARE default scaling coefficient: {scaling_coef}")
    
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
    
    # Step 1: Apply DARE processing to current task vector
    print(f"Applying DARE processing: dropping {drop_rate*100:.1f}% of parameters and rescaling")
    dare_processed = {}
    for key, tensor in current_vector_dict.items():
        if len(tensor.shape) >= 2:
            dare_processed[key] = apply_mask_and_rescale(tensor, drop_rate, mask_strategy)
        else:
            dare_processed[key] = tensor.clone()
    
    # Step 2: Handle vector merging logic based on task index
    if task_index == 0 or DARE_TASK_VECTOR is None:
        # First task, directly use processed vector as accumulated vector
        print("First task or accumulated vector not initialized, using current DARE-processed task vector")
        DARE_TASK_VECTOR = {k: v.clone() for k, v in dare_processed.items()}
    else:
        # Not the first task, accumulate to current vector
        print("Accumulating current DARE-processed task vector to accumulated vector")
        
        # Ensure all keys are handled
        all_keys = set(DARE_TASK_VECTOR.keys()).union(dare_processed.keys())
        
        for key in all_keys:
            if key in DARE_TASK_VECTOR and key in dare_processed:
                # Ensure data types are consistent
                if DARE_TASK_VECTOR[key].dtype != dare_processed[key].dtype:
                    dare_processed[key] = dare_processed[key].to(DARE_TASK_VECTOR[key].dtype)
                
                # Both vectors have this key, accumulate
                DARE_TASK_VECTOR[key] += dare_processed[key]
            elif key in dare_processed:
                # Only current vector has this key, add to accumulated vector
                DARE_TASK_VECTOR[key] = dare_processed[key].clone()
    
    # Create a copy of accumulated vector to avoid accidental modification of global variable
    combined_vector_dict = {k: v.clone() for k, v in DARE_TASK_VECTOR.items()}
    
    # Release unnecessary variables to save memory
    del dare_processed, current_vector_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Apply merged vector to base model and create final model
    print(f"Applying DARE accumulated task vector to base model (scaling_coef={scaling_coef})...")
    merged_model = apply_task_vector_dict(
        combined_vector_dict, base_model, scaling_coef=scaling_coef)
    
    # Release merged vector
    del combined_vector_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return merged_model

def save_dare_vector_to_disk(directory):
    """Save current DARE task vector to disk"""
    global DARE_TASK_VECTOR
    if DARE_TASK_VECTOR is not None:
        vector_path = os.path.join(directory, "dare_task_vector.pt")
        torch.save(DARE_TASK_VECTOR, vector_path)
        print(f"Saved DARE task vector to: {vector_path}")
        return vector_path
    return None

def load_dare_vector_from_disk(directory):
    """Load DARE task vector from disk"""
    global DARE_TASK_VECTOR
    vector_path = os.path.join(directory, "dare_task_vector.pt")
    if os.path.exists(vector_path):
        DARE_TASK_VECTOR = torch.load(vector_path)
        print(f"Loaded DARE task vector: {vector_path}")
        return True
    print(f"DARE task vector file not found: {vector_path}")
    return False