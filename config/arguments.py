import argparse
from utils.experiment_utils import DEFAULT_HF_TOKEN, DEFAULT_HF_CACHE


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Continual Model Merging - Using Pre-trained Model Fine-tuning and Multiple Merging Methods")

    parser.add_argument('--base-model', default='meta-llama/Llama-3.2-1B-Instruct', type=str,
                       help='Base model path')
    parser.add_argument('--save-path', default='./experiments/', type=str,
                        help='Merged model save path')
    parser.add_argument('--cache-dir', default=DEFAULT_HF_CACHE, type=str,
                        help='HuggingFace cache directory')
    parser.add_argument('--token', default=DEFAULT_HF_TOKEN, type=str,
                        help='HuggingFace API token')

    parser.add_argument('--task-names', default='C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten', type=str,
                        help='Comma-separated list of task names')
    parser.add_argument('--custom-datasets', action='store_true', default=True,
                        help='Use custom datasets (do not use MergeBench datasets)')

    parser.add_argument('--batch-size', default=8, type=int,
                        help='Fine-tuning batch size')
    parser.add_argument('--learning-rate', default=2e-5, type=float,
                        help='Learning rate')
    parser.add_argument('--epochs', default=5, type=int,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', default='mergopt', type=str,
                        choices=['adam', 'mergopt', 'sgd'],
                        help='Select optimizer type: adam (AdamW optimizer), mergopt (MergOPT optimizer), sgd (SGD optimizer)')
    
    parser.add_argument('--base-optimizer', default='adamw', type=str, 
                        choices=['adamw', 'sgd'],
                        help='Base optimizer type: adamw (AdamW optimizer), sgd (SGD optimizer)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum coefficient (only valid when base-optimizer=sgd)')
    parser.add_argument('--nesterov', default=False, action='store_true',
                        help='Use Nesterov momentum (only valid when base-optimizer=sgd)')
    
    parser.add_argument('--mergopt-mu', default=0, type=float,
                        help='MergOPT Laplace distribution location parameter (only valid when optimizer=mergopt)')
    parser.add_argument('--mergopt-b', default=0.0005, type=float,
                        help='MergOPT Laplace distribution scale parameter (only valid when optimizer=mergopt)')
    parser.add_argument('--mergopt-k-max', default=7, type=int,
                        help='MergOPT maximum number of tasks (only valid when optimizer=mergopt)')

    parser.add_argument('--task-vector-from-base', default=False, action='store_true',
                        help='Compute task vectors from base model (True: fine-tuned - base model, False: fine-tuned - current merged model)')
    parser.add_argument('--scaling-coef', default=1.0, type=float,
                        help='Merging scaling coefficient')
    parser.add_argument('--use-default-scaling', action='store_true', default=False,
                    help='Use algorithm default scaling coefficient, ignore --scaling-coef parameter value')
    parser.add_argument('--merge-method', default='task_arithmetic', type=str,
                        choices=['task_arithmetic','ties_merge', 'dare', 'swa'],
                        help='Merging method: task_arithmetic, ties_merge, dare or swa')

    parser.add_argument('--evaluate-at-end', action='store_true', default=False,
                    help='Whether to evaluate after all tasks are completed, default False')
    
    parser.add_argument('--start-task', default=0, type=int,
                        help='Starting task index (0-6)')
    parser.add_argument('--end-task', default=6, type=int,
                        help='Ending task index (0-6)')
    parser.add_argument('--continue-experiment', action='store_true',
                        help='Continue previous experiment')
    parser.add_argument('--prev-experiment-dir', default=None, type=str,
                        help='Directory of previous experiment')

    return parser.parse_args()