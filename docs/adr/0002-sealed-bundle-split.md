# ADR 0002: split inference and evaluation bundles

Decision: use an immutable optimizer-free `InferenceBundle`, a separately
content-addressed `EvaluationBundle`, and a `SealedRun` aggregate binding both
plus the claim audit.

This deliberately supersedes the amendment's monolithic final-inference
surface. It prevents evaluation code and one-shot access receipts from mutating
or being confused with portable inference weights, while the sealed aggregate
preserves one transferable identity.
