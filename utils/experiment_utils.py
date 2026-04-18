import json
import os
import datetime
from huggingface_hub import login
import torch

DEFAULT_HF_CACHE = None # Please replace with your own cache directory
DEFAULT_HF_TOKEN = None # Please replace with your own token

task_specific_epochs = {
    'C-STANCE': 5,
    'FOMC': 3,
    'MeetingBank': 7,
    'ScienceQA': 3,
    'NumGLUE-cm': 5,
    'NumGLUE-ds': 5,
    '20Minuten': 7
}
task_specific_batch_sizes = {
    'C-STANCE': 8,
    'FOMC': 8,
    'MeetingBank': 8,
    'ScienceQA': 8,
    'NumGLUE-cm': 8,
    'NumGLUE-ds': 8,
    '20Minuten': 8
}


def create_experiment_dir(base_path="./experiments/"):
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.join(base_path, f"continual_model_merging_{timestamp}")
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(dir_path, "finetunedModels"), exist_ok=True)
    return dir_path


def setup_hf_cache(cache_dir=None):
    cache_dir = cache_dir or DEFAULT_HF_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(cache_dir, 'hub')
    os.environ['XDG_CACHE_HOME'] = cache_dir

    os.environ['TRANSFORMERS_OFFLINE'] = "0"
    os.environ['PYTHONWARNINGS'] = "ignore::UserWarning"
    os.environ['PYTHONIOENCODING'] = "utf-8"
    os.environ['LC_ALL'] = "C.UTF-8"
    os.environ['LANG'] = "C.UTF-8"
    print(f"HuggingFace cache directory set: {cache_dir}")


def huggingface_login(token=None):
    token = token or DEFAULT_HF_TOKEN

    try:
        print("Logging in to HuggingFace...")
        os.makedirs(os.path.expanduser("~/.huggingface"), exist_ok=True)
        with open(os.path.expanduser("~/.huggingface/token"), "w") as f:
            f.write(token)
        os.chmod(os.path.expanduser("~/.huggingface/token"), 0o600)
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
        login(token=token)
        print("HuggingFace login successful")
        return True
    except Exception as e:
        print(f"HuggingFace login failed: {e}")
        return False


def save_model_state_to_disk(state_dict, directory):
    state_path = os.path.join(directory, "previous_model_state.pt")
    print(f"Saving model state to: {state_path}")
    torch.save(state_dict, state_path)
    return state_path


def load_model_state_from_disk(directory):
    state_path = os.path.join(directory, "previous_model_state.pt")
    if os.path.exists(state_path):
        print(f"Loading model state from disk: {state_path}")
        return torch.load(state_path, map_location="cpu")
    return None


def save_merge_config(result_dir, merge_method, base_model, task_names,
                      start_task, end_task, scaling_coef=1.0, use_default_scaling=False,
                      task_vector_from_base=True,
                      continue_experiment=False, prev_experiment_dir=None,
                      final_model_only=False):
    config_path = os.path.join(result_dir, "merge_config.json")
    if not os.path.exists(config_path) or not continue_experiment:
        default_scaling = {
            "task_arithmetic": 0.2,
            "ties_merge": 0.2,
            "dare": 0.2,
            "swa": "NULL"
        }
        actual_scaling = default_scaling.get(
            merge_method, 1.0) if use_default_scaling else scaling_coef

        config_info = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": base_model,
            "tasks": task_names[start_task:end_task+1],
            "task_vector_from_base": task_vector_from_base,
            "merge_method": merge_method,
            "merge_params": {
                "scaling_coef": actual_scaling,
                "use_default_scaling": use_default_scaling
            },
            "continue_from": prev_experiment_dir if continue_experiment else None,
            "final_model_only": final_model_only
        }

        with open(config_path, "w") as f:
            json.dump(config_info, f, indent=2)
        print(f"Merge config saved to: {config_path}")

    return config_path


def restore_merge_state(merge_method, result_dir, prev_task_idx, previous_model_state, cache_dir):
    if prev_task_idx < 0:
        print("No previous task, no need to restore state")
        return False

    success = False

    if merge_method == 'task_arithmetic':
        print("Attempting to load cumulative task vector...")
        from merging.task_arithmetic import load_cumulative_vector_from_disk
        loaded = load_cumulative_vector_from_disk(result_dir)
        if loaded:
            print("Successfully loaded cumulative task vector")
            success = True
        else:
            print("Cumulative task vector not found, will start accumulation from beginning")

    elif merge_method == 'ties_merge':
        print("Attempting to load TIES cumulative task vector...")
        from merging.ties_merge import load_ties_vector_from_disk
        loaded = load_ties_vector_from_disk(result_dir)
        if loaded:
            print("Successfully loaded TIES cumulative task vector")
            success = True
        else:
            print("TIES cumulative task vector not found, will start accumulation from beginning")

    elif merge_method == 'dare':
        print("Attempting to load DARE cumulative task vector...")
        from merging.dare import load_dare_vector_from_disk
        loaded = load_dare_vector_from_disk(result_dir)
        if loaded:
            print("Successfully loaded DARE cumulative task vector")
            success = True
        else:
            print("DARE cumulative task vector not found, will start accumulation from beginning")
    elif merge_method == 'swa':
        print("Attempting to load SWA cumulative state...")
        from merging.swa import load_swa_state_from_disk
        loaded = load_swa_state_from_disk(result_dir)
        if loaded:
            print("Successfully loaded SWA cumulative state")
            success = True
        else:
            print("SWA cumulative state not found, will start accumulation from beginning")
    else:
        print(f"Unknown merge method: {merge_method}")

    return success