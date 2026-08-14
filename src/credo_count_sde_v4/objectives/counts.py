"""Exact complete-denominator Dirichlet–multinomial objective."""

from __future__ import annotations

import torch


def count_probabilities(
    source_counts: torch.Tensor,
    raw_fitness: torch.Tensor,
    duration: torch.Tensor,
    *,
    source_smoothing: float = 0.5,
) -> torch.Tensor:
    """Normalize every declared category in one complete denominator block."""

    if source_smoothing != 0.5:
        raise ValueError("v4.0 source smoothing is frozen at 0.5.")
    exposure = source_counts.to(raw_fitness.dtype) + source_smoothing
    return torch.softmax(torch.log(exposure) + duration * raw_fitness, dim=-1)


def dirichlet_multinomial_log_prob(
    counts: torch.Tensor,
    logits: torch.Tensor,
    log_concentration: torch.Tensor,
) -> torch.Tensor:
    """Return exact full-category DM log probability in float64.

    ``log_concentration`` is a historical parameter name; it is transformed
    exactly once through softplus to ensure strict positivity.
    """

    y = counts.to(torch.float64)
    eta = logits.to(torch.float64)
    concentration = torch.nn.functional.softplus(log_concentration.to(torch.float64))
    probabilities = torch.softmax(eta, dim=-1)
    alpha = concentration * probabilities
    total = y.sum(dim=-1)
    value = torch.lgamma(total + 1.0) - torch.lgamma(y + 1.0).sum(dim=-1)
    value = value + torch.lgamma(concentration) - torch.lgamma(total + concentration)
    value = value + (torch.lgamma(y + alpha) - torch.lgamma(alpha)).sum(dim=-1)
    return value


def exact_count_loss(
    terminal_counts: torch.Tensor,
    source_counts: torch.Tensor,
    raw_fitness: torch.Tensor,
    duration: torch.Tensor,
    pool_index: torch.Tensor,
    log_concentration: torch.Tensor,
    *,
    source_smoothing: float = 0.5,
) -> torch.Tensor:
    """Average negative DM log probability over complete physical pools."""

    if source_smoothing != 0.5:
        raise ValueError("v4.0 source smoothing is frozen at 0.5.")
    losses: list[torch.Tensor] = []
    for pool in torch.unique(pool_index):
        mask = pool_index == pool
        exposure = source_counts[mask].to(raw_fitness.dtype) + source_smoothing
        logits = torch.log(exposure) + duration[mask] * raw_fitness[mask]
        losses.append(
            -dirichlet_multinomial_log_prob(terminal_counts[mask], logits, log_concentration[pool])
        )
    return torch.stack(losses).mean().to(raw_fitness.dtype)
