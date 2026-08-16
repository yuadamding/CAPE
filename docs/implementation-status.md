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
- restartable particle-state integration and stable absolute log-mass pools;
- CLI/API, full content/reload verification, correction/context audit helpers;
- CPU regression, leakage, corruption, fault, resume, compatibility, and wheel CI.

Dev25 retains the component-qualified boundary introduced in dev18: the T00
pooled finite-measure contract, public API/CLI, strict channel-isolation
contract, and focused invariants. The external pooled Renz adapter passed T00
with 495 guides and 277,200 cells. T04 independently passed its fixed-truth
numerical qualification. T02A now freezes the independent raw-count and
relative-mass sampling noise surface from 100 repeats without a learned model.
T07S-A separately passes learned constant target-average reaction recovery on
synthetic complete-denominator catalogs after the duration-correct dev25
unified R0/R1 amendment; no pooled Renz dynamics component is promoted. See
[component qualification](component-qualification.md).

T01 v1 and v2 are implemented fail-closed and have been exercised on the
pooled Renz population. Both correctly retained the global decoder in all four
folds because every low-rank candidate worsened source-only count likelihood.
The raw Hellinger candidates were 0.204–0.256 nats/count worse; centered
residual Hellinger candidates were 0.178–0.238 worse. These are retired
candidates, not a representation qualification; downstream tests remain
blocked only on the pooled real-data representation path. The independent
T02A raw-count/mass calibration passes with a target-balanced observed-endpoint
sampling RMSE q95 of 0.132578 and guide absolute-error q95 target-median q95 of
0.289973. These are conditional noise summaries, not a direct model-improvement
margin or formal minimum detectable effect. Synthetic T04–T07 remains independent. See the
[detailed T02A record](t02a-raw-count-mass-noise.md).

T04 routes non-trainable analytic truths through the production streaming
Euler–Maruyama engine. It qualifies deterministic drift, OU moments over a
16-point particle/step grid with 50 seeds, exact reaction mass, stabilized
absolute-weight pool feedback, normalized-context negative control,
deterministic replay, and interrupted/resumed parity. The largest-grid OU
variance relative error is 0.004459. This is numerical software evidence; it
does not qualify a learned channel or biological model. See the
[detailed T04 record](t04-particle-engine-qualification.md).

T07S-A isolates target constant reaction as the sole trainable channel. It uses
119 independent zero-truth refits split into calibration and audit sets,
retains update 0, selects on separate validation catalogs, performs a fresh
post-selection refit, and evaluates independent test catalogs once. Dev25
preserves that selected model byte-for-byte and corrects both R0 and R1 to
duration-integrated centered interval change. The R1 target-balanced RMSE
is 0.033191 versus 0.706508 for zero reaction; delta -0.673317 with target-
bootstrap interval [-0.829965, -0.482187]. Reaction RMSE is 0.017233 and all
12 nonzero signs are recovered. Fixed channels, the complete-denominator gauge, probability sums,
control mask, and streaming relative masses pass their tolerances. This is
software/synthetic qualification, not evidence about a cohort. See the
[dev23 T07S record](t07s-reaction-recovery.md) and
[dev24 R1 amendment](t07s-reaction-metric-amendment.md) and authoritative
[dev25 unified R0/R1 amendment](t07s-null-interval-amendment.md).

Local acceptance result: 134 tests passed with one CUDA-only skip and 86.56%
coverage; `receipts/local-validation.json` is the machine authority. Ruff,
mypy, generated-schema, documentation-link, wheel
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

Deliberately not asserted complete in `4.0.0.dev25`:

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
dev25 commit and release receipt are necessary—but not sufficient—before any
pilot deployment or stable promotion.
