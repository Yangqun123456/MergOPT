import numpy as np
import torch


class MergOPT(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, mu=0, b=0.01, k_max=7,
                 lr=2e-5, weight_decay=0, momentum=0.9, nesterov=False, **kwargs):
        defaults = dict(
            lr=lr,
            mu=mu,
            b=b,
            k_max=k_max,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            **kwargs,
        )
        super(MergOPT, self).__init__(params, defaults)

        self.base_optimizer_class = base_optimizer
        self.param_groups_backup = []
        self.alpha_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.base_optimizers = []

        for param_group in self.param_groups:
            self.param_groups_backup.append([p.clone().detach() for p in param_group['params']])
            if base_optimizer == torch.optim.SGD:
                self.base_optimizers.append(
                    self.base_optimizer_class(
                        param_group['params'],
                        lr=param_group['lr'],
                        momentum=param_group['momentum'],
                        nesterov=param_group['nesterov'],
                        weight_decay=param_group['weight_decay'],
                        **kwargs,
                    )
                )
            else:
                self.base_optimizers.append(
                    self.base_optimizer_class(
                        param_group['params'],
                        lr=param_group['lr'],
                        weight_decay=param_group['weight_decay'],
                        **kwargs,
                    )
                )

    def sample_laplace(self, shape, mu, b, device):
        u = torch.rand(shape, device=device)
        return mu - b * torch.sign(u - 0.5) * torch.log(1 - 2 * torch.abs(u - 0.5))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            mu = group['mu']
            b = group['b']
            k_max = group['k_max']
            alpha = np.random.choice(self.alpha_set)
            task_count = np.random.randint(1, k_max + 1)
            perturbation_coef = task_count * alpha - 1

            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if param_idx >= len(self.param_groups_backup[group_idx]):
                    self.param_groups_backup[group_idx].append(p.clone().detach())
                else:
                    self.param_groups_backup[group_idx][param_idx].copy_(p.data)
                z = self.sample_laplace(p.shape, mu, b, p.device)
                p.data.add_(perturbation_coef * z)

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                p.data.copy_(self.param_groups_backup[group_idx][param_idx])

        for optimizer in self.base_optimizers:
            optimizer.step()

        return loss