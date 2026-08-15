# Implementation status

Last verified: 2026-08-15. Authority: package engineering status; biological
run receipts and cohort audits remain external to this repository.

Implemented in this repository:

- independent package/repository, licenses, locks, frozen dependency receipt;
- strict contracts and generated schemas;
- canonical hashes, sparse CSR store, correction gate, representation cache;
- compiled run identity, state/measure/context model, exact full-block count loss;
- update-based training, safe optimizer serialization, checkpoint/resume/fork;
- physical-grid streaming inference, latent-state export, four contrasts;
- one-shot evaluation, baseline audit, lifecycle ledger, sealed aggregate;
- CLI/API, full content/reload verification, correction/context audit helpers;
- CPU regression, leakage, corruption, fault, resume, compatibility, and wheel CI.

Dev18 adds the first component-qualified boundary: the T00 pooled
finite-measure contract, public API/CLI, strict channel-isolation contract, and
eight focused invariants. The external pooled Renz adapter passed T00 with 495
guides and 277,200 cells, but no T01–T13 component is promoted. See
[component qualification](component-qualification.md).

T01 v1 is implemented fail-closed and has been exercised on the pooled Renz
population. It correctly retained the global decoder in all four folds because
every low-rank candidate worsened source-only count likelihood. This is a
retired candidate, not a representation qualification; downstream tests remain
blocked.

Local acceptance result: see `receipts/local-validation.json` for the current
test count and coverage; one CUDA
test was skipped because the validation environment exposed no CUDA device. Ruff,
generated-schema, documentation-link, wheel
namespace/license, clean-wheel lifecycle, and frozen-CREDO preflight checks
passed. The wheel is reproducible under a fixed `SOURCE_DATE_EPOCH`; normalized
sdist publication is reproducible through `scripts/normalize_sdist.py`.

Dev7 replaced the invalid reconstructed-SVD softmax with a checkpoint-owned
multinomial count-composition decoder. Dev10 added a deterministic
training-only validation holdout and minimum-validation-cross-entropy
checkpoint selection. Dev12 caches the immutable sparse decoder row universe
once per training process, eliminating repeated whole-window HDF5 scans while
preserving row order, targets, validation rows, and checkpoint semantics.
Dev13 adds an analytically fitted adaptive target anchor around a global
training-terminal reference. It uses support-weighted leave-one-guide-out
sister-guide evidence, a bounded nonnegative gate, and exact-zero control or
discordant-target corrections. That state estimator is frozen before decoder
optimization; CUDA updates do not train it.
Decoder reconstruction validation does not establish perturbation prediction
or biological validity.

Dev14 adds a null-nested source-conditioned terminal-anchor residual and an
explicit joint state/decoder training switch. It replaces atomic CUDA decoder
reductions with deterministic CSR segment reductions and applies the frozen
primary-population definition consistently to both point metrics and target
bootstraps. Local acceptance establishes contracts and CPU behavior; the
completed dev14 H100 run below establishes the external development result.

Dev17 superseded dev16 before cohort execution. It estimates a bounded scalar
target-main shrinkage model from leave-one-guide-out sister-target predictions
without an interaction, freezes that model, and
tests the source × target term only through the full-versus-interaction-off
increment. The selector uses separate target and interaction margins from a
byte-verified row-level calibration with at least 59 genuine independent null
fits. It admits exactly update 0 and the calibrated checkpoint schedule, and
records all three scores and displacement magnitudes. A typed
selection manifest is embedded in inference and reported by evaluation; only a
deployed interaction family that beats both M1 and M0 can pass the interaction
advancement gate. The
gene decoder is structurally absent. Post-selection refit publishes through
`REFIT_PLANNED → REFIT_RUNNING → REFIT_COMMITTED`, resumes after interruption,
uses no false state-parent edge, and records empirical and complete regularized
objectives separately. Local
acceptance covers global-only, shrunk-target, full-target, and interaction truth
regimes plus a 59-fit evidence-bearing null-calibration lifecycle. The pilot
also forbids diffusion, selection, alternative drift, analytic fitting,
decoder training, support weighting, and checkpoint forks. No real-cohort
dev15, dev16, or dev17 performance result exists. Dev18 further target-centers
the interaction, enforces target-wise zero mean, separates target-main,
conditional-interaction, and joint nulls, freezes fold and seed identities,
and strengthens noninteraction comparators. That T03 code remains unqualified
until T01, T02, and the prescribed pooled pilot pass.

Deliberately not asserted complete in `4.0.0.dev18`:

- stable CREDO entry-point discovery;
- in-repository real-cohort adapters, biological thresholds, or biological claims;
- claim-eligible context status without all external audit receipts;
- CUDA/BF16 hardware qualification or production numerical tolerance;
- stable legacy-family replay, SBOM attestation, and signed release archive.

Those are release gates, not silent assumptions. The current implementation is
a complete non-device engineering lifecycle and remains `engineering_only`.

Earlier external GSE235325 CUDA reruns failed their scientific baselines and
are retained as engineering diagnostics. The eight-worker dev12 H100
development run completed 2,000 decoder updates in all eight scope-fold jobs,
then finalized, CPU-evaluated, fully verified, and sealed every bundle. Sampled
peak execution memory was 19.29-20.44 decimal GB. Its state model was fixed to
the global training-terminal centroid; long optimization trained the decoder,
not an SDE trajectory. Final target-balanced state RMSE differed from that null
by -5.92e-09 (`limited2500`) and +4.99e-09 (`common34699`), so both scientific
advancement gates were false. The external final report and receipts—not this
package-status page—are the run authority.

The dev13/a10 successor completed all eight 2,000-update H100 workers and was
subsequently evaluated and sealed by a fresh Kubernetes recovery. It remained
non-promotable: `limited2500` improved by only 0.000922 RMSE and
`common34699` worsened by 0.001425 relative to the global-terminal comparator;
both conditional target-bootstrap intervals crossed zero. Those CUDA updates
trained only the decoder because the adaptive state estimator was analytic and
frozen. Exact run receipts remain the authority; these numbers are a package
status synopsis, not a biological result.

The dev14/a11 successor trained a zero-initialized source-conditioned residual
jointly with the decoder, so this state channel was no longer analytic or
detached. All eight projects completed 2,000 H100 updates and selected update
250 by training-only target-stratified state validation. The residual was
nonzero in every selected checkpoint, but outer performance was worse than the
global-terminal null: `limited2500` delta +0.012688, 95% conditional
target-bootstrap interval [+0.005258, +0.019539]; `common34699` delta
+0.009976, interval [+0.006061, +0.014019]. Both advancement gates failed.
This falsifies the tested source-state-only residual under the current
historically exposed development contract; additional epochs, particles, or
memory are not a justified retry. Exact external run receipts remain the
authority.

Exact future use depends on the branch commit, `REPOSITORY.sha256`, the
wheel/sdist bytes, generated schemas, and the local validation receipt. A clean
dev18 commit and release receipt are necessary—but not sufficient—before any
pilot deployment or stable promotion.
