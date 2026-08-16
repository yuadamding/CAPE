"""Cross-contract validation for the G14 evidence-only stage."""

from __future__ import annotations

from ..contracts import (
    ClaimRegistry,
    G14MultiplicityContract,
    G14RobustnessPlan,
    G14SealContract,
)
from ..errors import IntegrityError


def validate_g14_contracts(
    registry: ClaimRegistry,
    robustness: G14RobustnessPlan,
    multiplicity: G14MultiplicityContract,
    seal: G14SealContract,
) -> None:
    """Fail closed unless registry, sensitivity, multiplicity, and seal agree."""

    if multiplicity.claim_registry_id != registry.registry_id:
        raise IntegrityError("G14 multiplicity references a different claim registry.")
    if (
        seal.claim_registry_id != registry.registry_id
        or seal.robustness_plan_id != robustness.plan_id
        or seal.multiplicity_contract_id != multiplicity.multiplicity_contract_id
    ):
        raise IntegrityError("G14 seal references an incompatible contract graph.")
    records = {record.claim_id: record for record in registry.records}
    covered: list[str] = []
    for family in multiplicity.families:
        for claim_id in family.claim_ids:
            if claim_id not in records:
                raise IntegrityError(f"G14 multiplicity references unknown claim {claim_id}.")
            if records[claim_id].claim_family != family.family_id:
                raise IntegrityError(f"G14 claim {claim_id} occurs in the wrong family.")
            covered.append(claim_id)
    if set(covered) != set(records) or len(covered) != len(records):
        raise IntegrityError("G14 multiplicity must cover every claim exactly once.")
