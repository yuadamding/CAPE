"""Fail-closed public exception hierarchy."""


class CredoV4Error(RuntimeError):
    """Base v4 failure."""


class ContractError(CredoV4Error):
    """A strict contract is invalid or inconsistent."""


class IntegrityError(CredoV4Error):
    """Artifact bytes disagree with their declared identity."""


class CapabilityError(CredoV4Error):
    """The compiled run cannot perform the requested operation."""


class ResumeMismatchError(CredoV4Error):
    """A training-affecting setting changed across resume."""


class StateTransitionError(CredoV4Error):
    """The lifecycle transition is illegal."""
