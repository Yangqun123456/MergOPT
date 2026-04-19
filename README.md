# 【ICLR 2026】MergOPT: A Merge-Aware Optimizer for Robust Model Merging

## Abstract

> Model merging aims to integrate multiple independently fine-tuned expert models into a single model while preserving the knowledge of all experts. However, existing approaches mainly address parameter conflicts at the merging stage and overlook the role of the fine-tuning process, which often leads to significant post-merge performance degradation. To address this limitation, we propose a novel **merging-aware optimizer** (abbreviated as `MergOPT`) that injects principled merge-induced parameter shifts into the weight update steps so that the fine-tuned model exhibits a more stable loss landscape under subsequent merging operations. Specifically, we first formulate model merging as a distributionally robust optimization problem in the weight space: the parameters of other experts to be merged are viewed as adversarial merge-offsets, and fine-tuning adapts to the worst-case merging scenario. Building on this formulation, we analyze the distribution of parameter updates and the effects of merging hyperparameters, from which we derive a merging-guided feasible region for weight shifts. Finally, extensive experiments across four large language models (LLMs) and one vision model show that our approach consistently outperforms standard fine-tuning, yielding an average relative gain of 3.5% and a maximum gain of 9.5% across four merging strategies when merging seven experts.

## Citation
If you find our paper or this resource helpful, please consider cite:
```
@article{MergOPT_ICLR_2026,
  title={MergOPT: A Merge-Aware Optimizer for Robust Model Merging},
  author={Yang, Enneng and Yang, Qun and Wang, Peng and Tang, Anke and Guo, Guibing and Cao, Xiaochun and Shen, Li},
  booktitle={The Fourteenth International Conference on Learning Representations}
  year={2026}
}
```
Thanks!


## Run

- Main workflow: `python model_merging.py`
