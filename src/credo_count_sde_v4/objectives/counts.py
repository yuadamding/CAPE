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


def dirichlet_multinomial_log_prob_from_alpha(
    counts: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Return a DM log probability from an explicit positive alpha vector.

    This form is required for conditioning a physical-pool DM on the observed
    total of a guide subset.  Unlike :func:`dirichlet_multinomial_log_prob`, it
    does not renormalize that subset or reset its concentration.
    """

    y = counts.to(torch.float64)
    alpha64 = alpha.to(torch.float64)
    if bool(torch.any(alpha64 <= 0)):
        raise ValueError("Every Dirichlet concentration must be strictly positive.")
    concentration = alpha64.sum(dim=-1)
    total = y.sum(dim=-1)
    value = torch.lgamma(total + 1.0) - torch.lgamma(y + 1.0).sum(dim=-1)
    value = value + torch.lgamma(concentration) - torch.lgamma(total + concentration)
    value = value + (torch.lgamma(y + alpha64) - torch.lgamma(alpha64)).sum(dim=-1)
    return value


def conditional_dirichlet_multinomial_log_prob(
    counts_active: torch.Tensor,
    full_probabilities: torch.Tensor,
    active_mask: torch.Tensor,
    full_concentration: torch.Tensor | float,
) -> torch.Tensor:
    """Score an active subcomposition under one complete physical-pool DM.

    The complete catalog determines ``p`` and ``alpha = kappa * p``.  Only the
    terminal counts selected by ``active_mask`` are read.  The inherited
    conditional concentration is therefore ``sum(alpha[active_mask])`` rather
    than a fresh copy of ``kappa``.
    """

    probabilities = full_probabilities.to(torch.float64)
    mask = active_mask.to(torch.bool)
    if probabilities.ndim != 1 or mask.ndim != 1 or len(probabilities) != len(mask):
        raise ValueError("Physical-pool probabilities and active mask must be aligned vectors.")
    if int(mask.sum()) != int(counts_active.numel()):
        raise ValueError("Active counts do not match the physical-pool category mask.")
    if not torch.isclose(
        probabilities.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Physical-pool probabilities must sum to one.")
    concentration = torch.as_tensor(full_concentration, dtype=torch.float64)
    if concentration.numel() != 1 or float(concentration) <= 0:
        raise ValueError("Full physical-pool concentration must be a positive scalar.")
    return dirichlet_multinomial_log_prob_from_alpha(
        counts_active,
        concentration * probabilities[mask],
    )


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
