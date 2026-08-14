# Changelog

## 4.0.0.dev14 — 2026-08-13

- add a source-conditioned terminal-anchor residual that is initialized at the
  exact global-terminal null and selected by a training-only target-stratified
  validation split;
- make joint state/decoder optimization explicit so decoder workloads cannot
  silently detach the scientific state objective;
- replace CUDA atomic decoder reductions with deterministic CSR segment
  reductions and enforce deterministic-algorithm mode when requested;
- apply the frozen primary-population definition consistently to evaluation
  metrics and conditional target bootstraps.

## 4.0.0.dev13 — 2026-08-13

- replaced the Renz fixed global-terminal state null with an optional
  training-only, leave-one-guide-out adaptive target-anchor estimator;
- added target-balanced, support-aware state fitting and deterministic
  target-stratified inner-validation checkpoint selection;
- excluded one-update capacity checkpoints from scientific model selection;
- preserved the terminal null and all prior channels as explicit ablations.

## 4.0.0.dev12 — 2026-08-13

- Cache the immutable sparse gene-decoder row universe once per training process.
- Eliminate repeated full-window HDF5 scans during long random-minibatch training.
- Preserve exact row order, count targets, validation rows, and checkpoint semantics.

## 4.0.0.dev11 — 2026-08-13

- Added a fail-closed pure-Python frozen-CREDO verifier for pinned worker images without Git.
- The fallback verifies the exact commit reference, archive digest, checkout inventory, and every source byte.

## 4.0.0.dev10 — 2026-08-13

- Added a deterministic training-only gene-decoder holdout.
- Scores every persisted checkpoint by validation cross-entropy and binds the selected checkpoint.
- Preserves full-budget training and resumable checkpoint generations while preventing final-update-only selection.

## 4.0.0.dev9 — 2026-08-13

- Make environment identities portable across Kubernetes nodes by binding the
  execution image digest, Python ABI, package versions, CUDA runtime, and cuDNN
  while excluding node-specific kernel text.

## 4.0.0.dev8 — 2026-08-13

- Freeze the analytic state anchor while fitting the count-composition decoder.

## 4.0.0.dev7 — 2026-08-13

- Add a trained count-composition decoder with sparse multinomial targets.
- Enable large real-cell CUDA training batches without using CUDA for preprocessing/evaluation.
- Replace the invalid SVD-coordinate softmax decoder with checkpoint-owned logits.
- Add held-out gene-composition diagnostics.

## 4.0.0.dev6 — 2026-08-13

- Require scientific improvement to exceed the configured numerical tolerance.

## 4.0.0.dev5 — 2026-08-13

- Bind evaluation to the compiled source manifest and logical baseline registry.
- Reject ordered feature contracts that differ from the CountStore feature hash.
- Correct conditional target bootstrapping to preserve target-balanced weighting.
- Separate numerical parity from scientific advancement and serialize eligibility truthfully.

## 4.0.0.dev1 — 2026-08-12

- Establish the independent v4 repository and immutable compatibility boundary.
- Implement strict contracts, sparse storage, preparation, all three intents,
  resumable training, inference, counterfactuals, evaluation, and sealing.
- Bind actual source/permutation manifests, hash the executing wheel when supplied,
  fingerprint the realized runtime, and fail closed on post-compile code/config drift.
- Disable the unvalidated SVD gene decoder, bound interleaved sparse reads, and
  encode only the frozen information-set row universe.
