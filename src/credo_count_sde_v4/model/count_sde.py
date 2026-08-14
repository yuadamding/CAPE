"""Separate state, measure, and context channels for v4."""

from __future__ import annotations

import torch
from torch import nn

from ..contracts import ModelConfig, RunIntent


class CountSDEModel(nn.Module):
    """Small auditable count-SDE parameterization.

    The model deliberately avoids a single residual embedding controlling every
    channel. Controls receive exact-zero target residuals by multiplication.
    """

    def __init__(self, config: ModelConfig, intent: RunIntent) -> None:
        super().__init__()
        self.config = config
        self.intent = intent
        d, targets, pools = config.state_dim, config.target_count, config.pool_count
        self.base_drift = nn.Parameter(
            torch.zeros(d), requires_grad=not config.freeze_closed_form_drift
        )
        self.target_drift = nn.Parameter(
            torch.zeros(targets, d), requires_grad=not config.freeze_closed_form_drift
        )
        self.terminal_anchor = nn.Parameter(
            torch.zeros(d), requires_grad=config.trainable_terminal_anchor
        )
        self.target_anchor_offset = nn.Parameter(
            torch.zeros(targets, d), requires_grad=config.trainable_target_anchor
        )
        self.register_buffer("target_anchor_gate", torch.zeros(targets))
        if config.source_conditioned_anchor:
            self.source_anchor_hidden: nn.Linear | None = nn.Linear(d, config.hidden_dim, bias=True)
            self.source_anchor_output: nn.Linear | None = nn.Linear(
                config.hidden_dim, d, bias=False
            )
            nn.init.normal_(self.source_anchor_hidden.weight, std=0.02)
            nn.init.zeros_(self.source_anchor_hidden.bias)
            # A zero output layer makes the new family contain the exact global
            # terminal-anchor null before any optimization.
            nn.init.zeros_(self.source_anchor_output.weight)
        else:
            self.source_anchor_hidden = None
            self.source_anchor_output = None
        self.state_drift_hidden = nn.Linear(d, config.hidden_dim, bias=True)
        self.state_drift_output = nn.Linear(config.hidden_dim, d, bias=False)
        self.state_drift_hidden.requires_grad_(config.state_dependent_drift)
        self.state_drift_output.requires_grad_(config.state_dependent_drift)
        nn.init.normal_(self.state_drift_hidden.weight, std=0.02)
        nn.init.zeros_(self.state_drift_hidden.bias)
        nn.init.zeros_(self.state_drift_output.weight)
        if config.gene_decoder_features:
            if config.gene_decoder_hidden_dim:
                self.gene_decoder = nn.Sequential(
                    nn.Linear(d, config.gene_decoder_hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.gene_decoder_hidden_dim, config.gene_decoder_features),
                )
            else:
                self.gene_decoder = nn.Linear(d, config.gene_decoder_features)
        else:
            self.gene_decoder = None
        self.log_diffusion = nn.Parameter(
            torch.full((d,), -3.0), requires_grad=config.shared_diffusion
        )
        self.target_selection = nn.Parameter(
            torch.zeros(targets, d), requires_grad=config.centered_selection
        )
        measure = intent in {RunIntent.COUNT_MEASURE, RunIntent.COUNT_CONTEXT}
        self.pool_reference_fitness = nn.Parameter(torch.zeros(pools), requires_grad=measure)
        self.target_fitness = nn.Parameter(torch.zeros(targets), requires_grad=measure)
        self.log_concentration = nn.Parameter(torch.full((pools,), 2.0), requires_grad=measure)
        self.guide_efficacy_logit = nn.Parameter(
            torch.zeros(targets), requires_grad=measure and config.source_efficacy_sensitivity
        )
        context = intent is RunIntent.COUNT_CONTEXT
        self.state_pool_projection = nn.Parameter(
            torch.zeros(d, config.context_rank), requires_grad=context
        )
        self.state_mass_context = nn.Parameter(
            torch.zeros(config.context_rank), requires_grad=context
        )
        self.fitness_pool_projection = nn.Parameter(
            torch.zeros(d, config.context_rank), requires_grad=context
        )
        self.fitness_mass_context = nn.Parameter(
            torch.zeros(config.context_rank), requires_grad=context
        )
        self.target_fitness_context = nn.Parameter(
            torch.zeros(targets, config.context_rank), requires_grad=context
        )
        self.context_to_state = nn.Parameter(
            torch.zeros(config.context_rank, d), requires_grad=context
        )
        if context:
            nn.init.normal_(self.state_pool_projection, std=0.02)
            nn.init.normal_(self.fitness_pool_projection, std=0.02)
            nn.init.normal_(self.target_fitness_context, std=0.02)
            nn.init.normal_(self.context_to_state, std=0.02)

    def _pool_mediator(
        self,
        pool_state_mean: torch.Tensor,
        pool_log_mass: torch.Tensor,
        *,
        channel: str,
    ) -> torch.Tensor:
        if channel == "state":
            return (
                pool_state_mean @ self.state_pool_projection
                + pool_log_mass[:, None] * self.state_mass_context
            )
        return (
            pool_state_mean @ self.fitness_pool_projection
            + pool_log_mass[:, None] * self.fitness_mass_context
        )

    @staticmethod
    def _mask(is_control: torch.Tensor, ndim: int) -> torch.Tensor:
        mask = (~is_control.bool()).to(dtype=torch.float32)
        while mask.ndim < ndim:
            mask = mask.unsqueeze(-1)
        return mask

    def drift(
        self,
        target_index: torch.Tensor,
        pool_index: torch.Tensor,
        is_control: torch.Tensor,
        *,
        effect_mode: str = "factual",
        context_mode: str = "source_fixed",
        pool_state_mean: torch.Tensor | None = None,
        pool_log_mass: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = self.base_drift.expand(target_index.shape[0], -1)
        if effect_mode == "factual":
            result = result + self.target_drift[target_index] * self._mask(is_control, 2)
        if self.intent is RunIntent.COUNT_CONTEXT and context_mode != "source_fixed":
            if pool_state_mean is None or pool_log_mass is None:
                raise ValueError("Dynamic context requires a complete physical-pool bank.")
            context = self._pool_mediator(pool_state_mean, pool_log_mass, channel="state")[
                pool_index
            ]
            result = result + context @ self.context_to_state
        if z is not None and z.ndim == 3:
            result = result[:, None, :].expand(-1, z.shape[1], -1)
        if z is not None and self.config.state_dependent_drift:
            result = result + self.state_drift_output(torch.tanh(self.state_drift_hidden(z)))
        return result

    def diffusion(self) -> torch.Tensor:
        return (
            torch.nn.functional.softplus(self.log_diffusion)
            if self.config.shared_diffusion
            else torch.zeros_like(self.log_diffusion)
        )

    def decode_logits(self, state: torch.Tensor) -> torch.Tensor:
        if self.gene_decoder is None:
            raise RuntimeError("This model has no trained gene decoder.")
        return self.gene_decoder(state)

    def anchor(
        self,
        target_index: torch.Tensor,
        is_control: torch.Tensor,
        *,
        effect_mode: str = "factual",
        source_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = self.terminal_anchor.expand(target_index.shape[0], -1)
        if effect_mode == "factual":
            result = result + self.target_anchor_offset[target_index] * self._mask(is_control, 2)
        if self.config.source_conditioned_anchor:
            if source_z is None:
                raise ValueError("A source-conditioned anchor requires the original source state.")
            assert self.source_anchor_hidden is not None
            assert self.source_anchor_output is not None
            residual = self.source_anchor_output(torch.tanh(self.source_anchor_hidden(source_z)))
            result = result + self.config.source_anchor_residual_scale * torch.tanh(residual)
        return result

    def state_step(
        self,
        z: torch.Tensor,
        dt: torch.Tensor,
        target_index: torch.Tensor,
        pool_index: torch.Tensor,
        is_control: torch.Tensor,
        *,
        total_steps: int,
        effect_mode: str = "factual",
        context_mode: str = "source_fixed",
        pool_state_mean: torch.Tensor | None = None,
        pool_log_mass: torch.Tensor | None = None,
        source_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.terminal_anchor_drift:
            anchor = self.anchor(
                target_index,
                is_control,
                effect_mode=effect_mode,
                source_z=source_z,
            )
            if z.ndim == 3:
                anchor = anchor[:, None, :]
            alpha = self.config.source_carryover_alpha
            step_carryover = 0.0 if alpha == 0.0 else alpha ** (1.0 / total_steps)
            return anchor + step_carryover * (z - anchor)
        drift = self.drift(
            target_index,
            pool_index,
            is_control,
            effect_mode=effect_mode,
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
            z=z,
        )
        while dt.ndim < z.ndim:
            dt = dt.unsqueeze(-1)
        return z + dt * drift

    def centered_selection(
        self,
        z: torch.Tensor,
        weights: torch.Tensor,
        target_index: torch.Tensor,
        is_control: torch.Tensor,
        *,
        effect_mode: str = "factual",
    ) -> torch.Tensor:
        if not self.config.centered_selection or effect_mode == "reference":
            return torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
        vector = self.target_selection[target_index] * self._mask(is_control, 2)
        raw = torch.einsum("bpd,bd->bp", z, vector)
        normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        mean = (normalized * raw).sum(dim=1, keepdim=True)
        return raw - mean

    def raw_fitness(
        self,
        target_index: torch.Tensor,
        pool_index: torch.Tensor,
        is_control: torch.Tensor,
        *,
        effect_mode: str = "factual",
        context_mode: str = "source_fixed",
        pool_state_mean: torch.Tensor | None = None,
        pool_log_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.intent is RunIntent.COUNT_STATE:
            return torch.zeros_like(target_index, dtype=self.base_drift.dtype)
        result = self.pool_reference_fitness[pool_index]
        if effect_mode == "factual":
            target = self.target_fitness[target_index]
            if self.config.source_efficacy_sensitivity:
                target = target * torch.sigmoid(self.guide_efficacy_logit[target_index]) * 2.0
            result = result + target * self._mask(is_control, 1)
        if self.intent is RunIntent.COUNT_CONTEXT and context_mode != "source_fixed":
            if pool_state_mean is None or pool_log_mass is None:
                raise ValueError("Dynamic context requires a complete physical-pool bank.")
            pool = self._pool_mediator(pool_state_mean, pool_log_mass, channel="fitness")[
                pool_index
            ]
            target = self.target_fitness_context[target_index] * self._mask(is_control, 2)
            payoff = (pool * target).sum(dim=-1)
            result = result + payoff
        return result

    def relative_fitness(
        self,
        target_index: torch.Tensor,
        pool_index: torch.Tensor,
        is_control: torch.Tensor,
        source_exposure: torch.Tensor,
        *,
        effect_mode: str = "factual",
        context_mode: str = "source_fixed",
        pool_state_mean: torch.Tensor | None = None,
        pool_log_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self.raw_fitness(
            target_index,
            pool_index,
            is_control,
            effect_mode=effect_mode,
            context_mode=context_mode,
            pool_state_mean=pool_state_mean,
            pool_log_mass=pool_log_mass,
        )
        relative = torch.empty_like(raw)
        for pool in torch.unique(pool_index):
            mask = pool_index == pool
            weights = source_exposure[mask].clamp_min(0)
            weights = weights / weights.sum().clamp_min(1e-12)
            relative[mask] = raw[mask] - (weights * raw[mask]).sum()
        return relative

    def predict_means(
        self,
        source_z: torch.Tensor,
        duration: torch.Tensor,
        target_index: torch.Tensor,
        pool_index: torch.Tensor,
        is_control: torch.Tensor,
    ) -> torch.Tensor:
        drift = self.drift(target_index, pool_index, is_control, z=source_z)
        return source_z + duration.unsqueeze(-1) * drift
