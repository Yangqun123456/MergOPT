import os
import torch
import gc
from transformers import AutoModelForCausalLM
from merging.model_utils import apply_task_vector_dict

# Global variable to store accumulated task vectors
TIES_TASK_VECTOR = None

def keep_topk_magnitude(vector_dict, k_percent=30):
    """
    Keep the top k% elements with the largest absolute values for each tensor, set the rest to 0
    
    Args:
        vector_dict: Task vector dictionary
        k_percent: Percentage of parameters to keep (1-100)
        
    Returns:
        Trimmed vector dictionary
    """
    if k_percent >= 100:
        return {key: value.clone() for key, value in vector_dict.items()}
    
    trimmed_dict = {}
    for key, tensor in vector_dict.items():
        # Only trim tensors with 2D or higher dimensions
        if len(tensor.shape) < 2:
            trimmed_dict[key] = tensor.clone()
            continue
            
        # Calculate the number of elements to keep
        numel = tensor.numel()
        num_keep = max(1, int(numel * k_percent / 100))
        
        # Get the absolute value threshold
        flat_tensor = tensor.reshape(-1)
        abs_tensor = torch.abs(flat_tensor)
        threshold = torch.kthvalue(abs_tensor, numel - num_keep + 1)[0]
        
        # Create mask
        mask = (abs_tensor >= threshold).reshape(tensor.shape)
        
        # Apply mask
        trimmed_dict[key] = tensor * mask
        
    return trimmed_dict

def ties_merge(base_model_path, current_vector_dict,
              task_index=None, task_count=1, scaling_coef=0.2,
              cache_dir="/root/autodl-tmp/huggingface", keep_percent=30):  
    """
    Merge models using TIES_Merge method - incremental implementation to reduce memory usage
    
    Principle: 1. Trim parameters with smaller absolute values in task vectors
               2. Directly accumulate and merge multiple task vectors
    
    Args:
        base_model_path: Base model path
        current_vector_dict: Pre-computed current task vector
        task_index: Current task index
        task_count: Total number of processed tasks
        scaling_coef: Task vector scaling coefficient
        cache_dir: Cache directory
        keep_percent: Percentage of parameters to keep during trimming, default 20%
        previous_model_state: Compatibility parameter, not used but kept for interface consistency
    
    Returns:
        model: Merged model
    """
    global TIES_TASK_VECTOR
    
    print(f"Performing TIES_Merge continuous merging, current task index: {task_index}, accumulated tasks: {task_count}")
    
    if scaling_coef is None:
        scaling_coef = 0.2
        print(f"Using TIES_Merge default scaling coefficient: {scaling_coef}")
    
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
    
    # Step 1: Trim redundant parameters in current task vector
    print(f"Trimming task vector, keeping top {keep_percent}% parameters by absolute value")
    trimmed_vector = keep_topk_magnitude(current_vector_dict, keep_percent)
    
    # Step 2: Handle vector merging logic based on task index
    if task_index == 0 or TIES_TASK_VECTOR is None:
        # First task, directly use trimmed vector as accumulated vector
        print("First task or accumulated vector not initialized, using current trimmed task vector")
        TIES_TASK_VECTOR = {k: v.clone() for k, v in trimmed_vector.items()}
    else:
        # Not the first task, directly accumulate trimmed vector
        print("Directly accumulating current trimmed task vector to accumulated vector")
        
        # Handle keys already in TIES_TASK_VECTOR
        for key in TIES_TASK_VECTOR:
            if key in trimmed_vector:
                # Ensure data types are consistent
                if TIES_TASK_VECTOR[key].dtype != trimmed_vector[key].dtype:
                    trimmed_vector[key] = trimmed_vector[key].to(TIES_TASK_VECTOR[key].dtype)
                
                # Directly accumulate
                TIES_TASK_VECTOR[key] += trimmed_vector[key]
        
        # Add keys not in TIES_TASK_VECTOR
        for key in trimmed_vector:
            if key not in TIES_TASK_VECTOR:
                TIES_TASK_VECTOR[key] = trimmed_vector[key].clone()
    
    # Create a copy of accumulated vector to avoid accidental modification of global variable
    combined_vector_dict = {k: v.clone() for k, v in TIES_TASK_VECTOR.items()}
    
    # Release unnecessary variables to save memory
    del trimmed_vector, current_vector_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Apply merged vector to base model and create final model
    print(f"Applying TIES accumulated task vector to base model (scaling_coef={scaling_coef})...")
    merged_model = apply_task_vector_dict(
        combined_vector_dict, base_model, scaling_coef=scaling_coef)
    
    # Release merged vector
    del combined_vector_dict
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return merged_model

def save_ties_vector_to_disk(directory):
    """Save current TIES task vector to disk"""
    global TIES_TASK_VECTOR
    if TIES_TASK_VECTOR is not None:
        vector_path = os.path.join(directory, "ties_task_vector.pt")
        torch.save(TIES_TASK_VECTOR, vector_path)
        print(f"Saved TIES task vector to: {vector_path}")
        return vector_path
    return None

def load_ties_vector_from_disk(directory):
    """Load TIES task vector from disk"""
    global TIES_TASK_VECTOR
    vector_path = os.path.join(directory, "ties_task_vector.pt")
    if os.path.exists(vector_path):
        TIES_TASK_VECTOR = torch.load(vector_path)
        print(f"Loaded TIES task vector: {vector_path}")
        return True
    print(f"TIES task vector file not found: {vector_path}")
    return False