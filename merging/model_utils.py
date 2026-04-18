import gc
import torch


def get_task_vector_dict(ft_model, base_model):
    """
    Compute task vectors for model parameters (difference between fine-tuned model and base model) - execute on CPU
    
    Args:
        ft_model: Fine-tuned model
        base_model: Base model
        
    Returns:
        dict: Task vector dictionary
    """
    task_vector_dict = {}
    ft_state_dict = ft_model.state_dict()
    base_state_dict = base_model.state_dict()
    
    # Explicitly use CPU device
    device = torch.device("cpu")
    print(f"Task vectors will be computed on device: {device}")
    
    # Record mismatched parameters
    mismatch_params = []
    skipped_params = []
    
    for key in base_state_dict.keys():
        if key in ft_state_dict:
            # Check if shapes match
            if ft_state_dict[key].shape != base_state_dict[key].shape:
                mismatch_params.append((key, ft_state_dict[key].shape, base_state_dict[key].shape))
                print(f"Warning: Parameter '{key}' shape mismatch - Fine-tuned model: {ft_state_dict[key].shape}, Base model: {base_state_dict[key].shape}")
                continue
                
            # Move both tensors to CPU
            base_param = base_state_dict[key].to(device)
            ft_param = ft_state_dict[key].to(device)
            
            # Compute difference on CPU
            task_vector_dict[key] = ft_param.float() - base_param.float()
        else:
            skipped_params.append(key)
    
    # Print compatibility report
    common_keys = set(base_state_dict.keys()).intersection(set(ft_state_dict.keys()))
    compatible_keys = set(task_vector_dict.keys())
    
    print(f"\nParameter compatibility report:")
    print(f"- Common parameters: {len(common_keys)}")
    print(f"- Shape-compatible parameters: {len(compatible_keys)}")
    print(f"- Shape-incompatible parameters: {len(mismatch_params)}")
    
    if len(compatible_keys) < len(common_keys) * 0.9:
        print("Warning: More than 10% of parameters are incompatible, model merging may not work well!")
    
    del ft_state_dict, base_state_dict, base_param, ft_param
    gc.collect()
    
    return task_vector_dict

def apply_task_vector_dict(task_vector_dict, base_model, scaling_coef=1.0):
    """
    Apply task vector to base model - execute on CPU
    
    Args:
        task_vector_dict: Task vector dictionary
        base_model: Base model
        scaling_coef: Scaling coefficient
        
    Returns:
        model: Model after applying task vector
    """
    # Specify execution on CPU
    cpu_device = torch.device("cpu")
    target_device = next(base_model.parameters()).device
    print(f"Applying task vector computed on CPU, final model will be moved to: {target_device}, scaling_coef={scaling_coef}")
    
    merged_state_dict = {}
    base_state_dict = base_model.state_dict()
    
    for key in base_state_dict.keys():
        if key in task_vector_dict:
            # Move base parameter to CPU
            base_param = base_state_dict[key].to(cpu_device)
            
            # Ensure task vector is on CPU
            task_vector = task_vector_dict[key]
            if task_vector.device != cpu_device:
                task_vector = task_vector.to(cpu_device)
                
            # Apply scaled task vector on CPU
            merged_param = base_param + (scaling_coef * task_vector)
            
            # Store merged result (keep on CPU)
            merged_state_dict[key] = merged_param

        else:
            # Copy other parameters directly, but move to CPU
            merged_state_dict[key] = base_state_dict[key].to(cpu_device)
        
    # Create new model instance (on CPU)
    merged_model = type(base_model)(base_model.config)
    
    # Load merged state dict
    merged_model.load_state_dict(merged_state_dict)
    
    # Move model back to original device (usually GPU)
    if str(target_device).startswith("cuda") and torch.cuda.is_available():
        print(f"Moving merged model back to device: {target_device}")
        merged_model = merged_model.to(target_device)
    
    del merged_state_dict, base_state_dict, base_param, task_vector, merged_param
    gc.collect()
    return merged_model