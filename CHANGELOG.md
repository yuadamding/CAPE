# Changelog

## 4.0.0.dev18 — 2026-08-15

- adopt a T00–T13 component-wise qualification ladder with fail-closed channel
  isolation and receipt-based downstream dependencies;
- add a cohort-neutral, order-invariant pooled finite-measure API and CLI with
  exact cell coverage, source-only eligibility, feature-order identity,
  complete control/target catalogs, Jeffreys-smoothed relative masses, and
  manifest-last publication;
- publish the standard component contract, metrics, activity, selection,
  receipt, and checksum surface even when null/bootstrap/model fields are not
  applicable;
- harden the optional T03 source × target pilot with training-target source
  centering, exact target-wise zero interaction, target-balanced full-batch
  optimization, separate two-null calibration, fixed split/seed identities,
  and strongest preregistered noninteraction comparators;
- prohibit scale-out and target-VRAM selection until prerequisite component
  receipts pass.
- add the fail-closed T01 count-native representation gate; the first external
  multinomial/Hellinger candidate was retired after dimension 0 won all folds.

## 4.0.0.dev17 — 2026-08-15

- replace the self-attested selection calibration with a verified row-level
  artifact containing at least 59 genuine independent null fits, exact seeds,
  protocol/configuration hashes, target multiplicity, support structure, and an
  exact one-sided zero-failure confidence bound;
- restrict null-guarded selection to update 0 plus the calibrated checkpoint
  grid, excluding an implicit final checkpoint;
- require the deployed interaction family to beat both its independent
  shrunk-target comparator and the global-terminal null under target-balanced
  bootstrap intervals and frozen margins;
- unify minibatch, complete-fit, and post-selection state objectives, align the
  analytic scalar ridge, freeze support weighting at zero for this pilot, and
  persist empirical and regularized objectives separately;
- preserve the generic non-interaction advancement gate and fail closed on
  diffusion, selection, alternative drift, analytic fit, decoder fields, and
  checkpoint forks in the interaction pilot;
- generate `REPOSITORY.sha256` exclusively from regular files in the Git index
  and verify it in clean-checkout CI.

## 4.0.0.dev16 — 2026-08-15

- separate target shrinkage from source × target learning by analytically
  fitting and freezing a bounded target-main coefficient before interaction
  optimization;
- score every interaction checkpoint against its own interaction-off ablation
  and require independent calibrated margins for target-only and interaction
  advancement;
- bind a typed selection manifest and selected family into inference and
  evaluation, which now fail closed for interaction claims when another family
  is deployed;
- prohibit every decoder architecture/training field in this pilot and derive
  decoder capability only from verified decoder training;
- make post-selection refit transactional and resumable, remove the false
  checkpoint-parent edge, and persist the exact refit objective;
- require a bound training-only calibration artifact and explicit early
  checkpoint schedule; normalize target interaction embeddings and use
  conventional ridge whitening.

## 4.0.0.dev15 — 2026-08-14

- add a rank-limited source × target terminal interaction with fit-only source
  ridge whitening, exact-zero controls, and explicit factual/reference
  semantics;
- persist the exact update-0 terminal-centroid null and add a null-guarded
  training-only selector with a frozen minimum-improvement margin;
- select explicitly among update 0, an information-matched sister-guide
  target-terminal model, and the target-plus-interaction model;
- refit the selected update budget from a fresh initialization on all
  outer-training state series before finalization;
- prohibit decoder training and require positive interaction shrinkage for
  this pilot channel;
- make CLI evaluation default explicitly to CPU, matching the public API and
  preventing an omitted `--device` from reaching `torch.device(None)`.

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
