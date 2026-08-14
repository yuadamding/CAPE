"""Typed dynamic physical-pool state with exact resume semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..errors import ContractError


@dataclass
class DynamicPoolBank:
    """Current source-allowed state and relative mass for every physical pool."""

    state_mean: torch.Tensor
    log_mass: torch.Tensor
    age: torch.Tensor
    contributor_count: torch.Tensor
    generation: int = 0

    @classmethod
    def from_series(
        cls,
        state: torch.Tensor,
        mass: torch.Tensor,
        pool_index: torch.Tensor,
        *,
        pool_count: int,
    ) -> DynamicPoolBank:
        means: list[torch.Tensor] = []
        totals: list[torch.Tensor] = []
        counts: list[int] = []
        for pool in range(pool_count):
            mask = pool_index == pool
            if not bool(mask.any()):
                raise ContractError(f"Physical pool {pool} has no declared contributor.")
            weights = mass[mask].clamp_min(0)
            normalized = weights / weights.sum().clamp_min(1e-12)
            means.append((normalized[:, None] * state[mask]).sum(dim=0))
            totals.append(weights.sum().log())
            counts.append(int(mask.sum()))
        return cls(
            state_mean=torch.stack(means),
            log_mass=torch.stack(totals),
            age=torch.zeros(pool_count, dtype=torch.int64, device=state.device),
            contributor_count=torch.tensor(counts, dtype=torch.int64, device=state.device),
        )

    def advance(self) -> None:
        self.age.add_(1)

    def refresh(
        self,
        state: torch.Tensor,
        mass: torch.Tensor,
        pool_index: torch.Tensor,
        *,
        protected_endpoint_accessed: bool = False,
    ) -> None:
        if protected_endpoint_accessed:
            raise ContractError("Protected endpoint observations cannot refresh a pool bank.")
        updated = type(self).from_series(
            state, mass, pool_index, pool_count=self.state_mean.shape[0]
        )
        self.state_mean.copy_(updated.state_mean)
        self.log_mass.copy_(updated.log_mass)
        self.contributor_count.copy_(updated.contributor_count)
        self.age.zero_()
        self.generation += 1

    def tensor_state(self) -> dict[str, torch.Tensor]:
        return {
            "state_mean": self.state_mean,
            "log_mass": self.log_mass,
            "age": self.age,
            "contributor_count": self.contributor_count,
            "generation": torch.tensor(self.generation, dtype=torch.int64),
        }

    @classmethod
    def from_tensor_state(cls, state: dict[str, torch.Tensor]) -> DynamicPoolBank:
        return cls(
            state_mean=state["state_mean"].clone(),
            log_mass=state["log_mass"].clone(),
            age=state["age"].clone(),
            contributor_count=state["contributor_count"].clone(),
            generation=int(state["generation"].item()),
        )
