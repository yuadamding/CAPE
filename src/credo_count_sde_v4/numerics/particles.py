"""Streaming Euler–Maruyama terminal rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model import CountSDEModel, DynamicPoolBank

ContextSchedule = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class ParticleState:
    """Minimal restart state for one streaming particle rollout."""

    z: torch.Tensor
    weights: torch.Tensor
    log_mass: torch.Tensor
    rng_state: torch.Tensor
    step: int

    def tensor_state(self) -> dict[str, torch.Tensor]:
        return {
            "z": self.z.clone(),
            "weights": self.weights.clone(),
            "log_mass": self.log_mass.clone(),
            "rng_state": self.rng_state.clone(),
            "step": torch.tensor(self.step, dtype=torch.int64),
        }

    @classmethod
    def from_tensor_state(cls, state: dict[str, torch.Tensor]) -> ParticleState:
        required = {"z", "weights", "log_mass", "rng_state", "step"}
        if set(state) != required:
            raise ValueError("Particle restart state has an incomplete tensor set.")
        return cls(
            z=state["z"].clone(),
            weights=state["weights"].clone(),
            log_mass=state["log_mass"].clone(),
            rng_state=state["rng_state"].clone(),
            step=int(state["step"].item()),
        )


def initialize_particle_state(
    source_z: torch.Tensor,
    *,
    particles: int,
    seed: int,
) -> ParticleState:
    """Initialize particles and a portable generator state without advancing."""

    if particles <= 0:
        raise ValueError("Particle count must be positive.")
    device, dtype = source_z.device, source_z.dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return ParticleState(
        z=source_z[:, None, :].expand(-1, particles, -1).clone(),
        weights=torch.full(
            (source_z.shape[0], particles),
            1.0 / particles,
            device=device,
            dtype=dtype,
        ),
        log_mass=torch.zeros(source_z.shape[0], device=device, dtype=dtype),
        rng_state=generator.get_state(),
        step=0,
    )


@torch.no_grad()
def advance_particle_state(
    model: CountSDEModel,
    state: ParticleState,
    source_z: torch.Tensor,
    duration: torch.Tensor,
    target_index: torch.Tensor,
    pool_index: torch.Tensor,
    is_control: torch.Tensor,
    source_exposure: torch.Tensor,
    *,
    total_steps: int,
    stop_step: int,
    effect_mode: str = "factual",
    context_mode: str = "source_fixed",
    context_schedule: ContextSchedule | None = None,
    capture_context: bool = False,
) -> tuple[ParticleState, ContextSchedule]:
    """Advance a restartable rollout to an absolute grid step."""

    if total_steps <= 0 or not state.step <= stop_step <= total_steps:
        raise ValueError("Particle restart step lies outside the integration grid.")
    if state.z.shape[0] != source_z.shape[0] or state.z.shape[2] != source_z.shape[1]:
        raise ValueError("Particle restart state shape differs from the source catalog.")
    if context_schedule is not None and len(context_schedule) != total_steps:
        raise ValueError("Frozen context schedule length differs from the integration grid.")
    generator = torch.Generator(device=source_z.device)
    generator.set_state(state.rng_state)
    z, weights, log_mass = state.z, state.weights, state.log_mass
    diffusion = model.diffusion()
    dt = duration / total_steps
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(state.step, stop_step):
        pool_state_mean: torch.Tensor | None = None
        pool_log_mass: torch.Tensor | None = None
        if context_mode != "source_fixed":
            if context_schedule is None:
                if torch.any(source_exposure < 0) or not bool(
                    torch.isfinite(source_exposure).all()
                ):
                    raise ValueError("Source exposure must be finite and nonnegative.")
                series_state = (z * weights.unsqueeze(-1)).sum(dim=1)
                source_log_mass = torch.where(
                    source_exposure > 0,
                    source_exposure.log(),
                    torch.full_like(source_exposure, -torch.inf),
                )
                bank = DynamicPoolBank.from_log_series(
                    series_state,
                    source_log_mass + log_mass,
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
            total_steps=total_steps,
            effect_mode=effect_mode,
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
            source_z=source_z,
        )
        if model.config.shared_diffusion:
            noise = torch.randn(
                z.shape,
                generator=generator,
                device=source_z.device,
                dtype=source_z.dtype,
            )
            z = z + torch.sqrt(dt)[:, None, None] * diffusion[None, None, :] * noise
        selection = model.centered_selection(
            z, weights, target_index, is_control, effect_mode=effect_mode
        )
        weights = weights * torch.exp(dt[:, None] * selection)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_mass = log_mass + dt * relative_fitness
    return (
        ParticleState(
            z=z,
            weights=weights,
            log_mass=log_mass,
            rng_state=generator.get_state(),
            step=stop_step,
        ),
        tuple(captured),
    )


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
    state = initialize_particle_state(source_z, particles=particles, seed=seed)
    state, captured = advance_particle_state(
        model,
        state,
        source_z,
        duration,
        target_index,
        pool_index,
        is_control,
        source_exposure,
        total_steps=steps,
        stop_step=steps,
        effect_mode=effect_mode,
        context_mode=context_mode,
        context_schedule=context_schedule,
        capture_context=capture_context,
    )
    return state.z, state.weights, torch.exp(state.log_mass), captured


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
