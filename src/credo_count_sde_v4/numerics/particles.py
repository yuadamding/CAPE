"""Streaming Euler–Maruyama terminal rollout."""

from __future__ import annotations

import torch

from ..model import CountSDEModel, DynamicPoolBank

ContextSchedule = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def _rollout(
    model: CountSDEModel,
    source_z: torch.Tensor,
    duration: torch.Tensor,
    target_index: torch.Tensor,
    pool_index: torch.Tensor,
    is_control: torch.Tensor,
    source_exposure: torch.Tensor,
    *,
    particles: int,
    steps: int,
    seed: int,
    effect_mode: str,
    context_mode: str,
    context_schedule: ContextSchedule | None,
    capture_context: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ContextSchedule]:
    device, dtype = source_z.device, source_z.dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = source_z[:, None, :].expand(-1, particles, -1).clone()
    weights = torch.full(
        (source_z.shape[0], particles), 1.0 / particles, device=device, dtype=dtype
    )
    log_mass = torch.zeros(source_z.shape[0], device=device, dtype=dtype)
    diffusion = model.diffusion()
    dt = duration / steps
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    if context_schedule is not None and len(context_schedule) != steps:
        raise ValueError("Frozen context schedule length differs from the integration grid.")
    for step in range(steps):
        pool_state_mean: torch.Tensor | None = None
        pool_log_mass: torch.Tensor | None = None
        if context_mode != "source_fixed":
            if context_schedule is None:
                series_state = (z * weights.unsqueeze(-1)).sum(dim=1)
                bank = DynamicPoolBank.from_series(
                    series_state,
                    source_exposure * torch.exp(log_mass),
                    pool_index,
                    pool_count=model.config.pool_count,
                )
                pool_state_mean, pool_log_mass = bank.state_mean, bank.log_mass
            else:
                pool_state_mean, pool_log_mass = context_schedule[step]
            if capture_context:
                captured.append((pool_state_mean.clone(), pool_log_mass.clone()))
        relative_fitness = model.relative_fitness(
            target_index,
            pool_index,
            is_control,
            source_exposure,
            effect_mode=effect_mode,
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
        )
        z = model.state_step(
            z,
            dt,
            target_index,
            pool_index,
            is_control,
            total_steps=steps,
            effect_mode=effect_mode,
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
            source_z=source_z,
        )
        if model.config.shared_diffusion:
            noise = torch.randn(z.shape, generator=generator, device=device, dtype=dtype)
            z = z + torch.sqrt(dt)[:, None, None] * diffusion[None, None, :] * noise
        selection = model.centered_selection(
            z, weights, target_index, is_control, effect_mode=effect_mode
        )
        weights = weights * torch.exp(dt[:, None] * selection)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_mass = log_mass + dt * relative_fitness
    return z, weights, torch.exp(log_mass), tuple(captured)


@torch.no_grad()
def rollout(
    model: CountSDEModel,
    source_z: torch.Tensor,
    duration: torch.Tensor,
    target_index: torch.Tensor,
    pool_index: torch.Tensor,
    is_control: torch.Tensor,
    source_exposure: torch.Tensor,
    *,
    particles: int,
    steps: int,
    seed: int,
    effect_mode: str = "factual",
    context_mode: str = "source_fixed",
    context_schedule: ContextSchedule | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain only current particles, weights, and relative mass."""

    z, weights, mass, _ = _rollout(
        model,
        source_z,
        duration,
        target_index,
        pool_index,
        is_control,
        source_exposure,
        particles=particles,
        steps=steps,
        seed=seed,
        effect_mode=effect_mode,
        context_mode=context_mode,
        context_schedule=context_schedule,
        capture_context=False,
    )
    return z, weights, mass


@torch.no_grad()
def rollout_with_context_schedule(
    model: CountSDEModel,
    source_z: torch.Tensor,
    duration: torch.Tensor,
    target_index: torch.Tensor,
    pool_index: torch.Tensor,
    is_control: torch.Tensor,
    source_exposure: torch.Tensor,
    *,
    particles: int,
    steps: int,
    seed: int,
    effect_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ContextSchedule]:
    """Run self-consistent pool feedback and retain only its bank schedule."""

    return _rollout(
        model,
        source_z,
        duration,
        target_index,
        pool_index,
        is_control,
        source_exposure,
        particles=particles,
        steps=steps,
        seed=seed,
        effect_mode=effect_mode,
        context_mode="pool_dynamic",
        context_schedule=None,
        capture_context=True,
    )
