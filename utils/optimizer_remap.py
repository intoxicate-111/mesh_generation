import torch
from typing import Optional


def rebuild_adam_optimizer(
    old_optimizer: torch.optim.Optimizer,
    new_parameters,
    lr_override: Optional[float] = None,
) -> torch.optim.Adam:
    if not old_optimizer.param_groups:
        raise ValueError("old_optimizer has no parameter groups")

    defaults = old_optimizer.defaults.copy()
    lr = defaults.get("lr", 1e-3) if lr_override is None else lr_override
    betas = defaults.get("betas", (0.9, 0.999))
    eps = defaults.get("eps", 1e-8)
    weight_decay = defaults.get("weight_decay", 0.0)
    amsgrad = defaults.get("amsgrad", False)

    return torch.optim.Adam(
        new_parameters,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )