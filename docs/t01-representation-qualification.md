# T01 count-native representation qualification

Last verified: 2026-08-15. Status: authoritative engineering record for the
T01 implementation in `4.0.0.dev19`; both real Renz candidates are retired.
This document defines software and development evidence, not a biological
claim.

## Decision

T01 did not pass. The exact global gene-frequency decoder, represented by
candidate dimension 0, won every outer development fold for both tested
count-native representation families. No learned encoder was therefore
selected, refit, or authorized for downstream use. T02 and every dynamics
stage remain blocked.

The result is a failure of the two tested Hellinger low-rank representation
families under the frozen T01 estimand. It is not evidence that count-native
representation learning is impossible, and it does not justify changing the
margin, dropping the null, or looking at P60 to select a replacement.

## Why T01 is separate

Representation learning can manufacture apparent trajectory predictability
if terminal cells, outer-held-out guides, or dynamics gradients influence the
encoder. T01 therefore qualifies an encoder before any drift, diffusion,
reaction, ecology, counterfactual, or gene-decoder component can run.

```mermaid
flowchart LR
    T00[Passed T00 pooled data] --> S[P4 training-guide counts]
    S --> I[P4-only inner split]
    I --> N[Dimension 0 global-frequency null]
    I --> C[Dimensions 8, 16, 32, 48]
    N --> D{Candidate beats null by frozen margin?}
    C --> D
    D -- No --> R[fail_retired; no refit; T02 blocked]
    D -- Yes in every fold --> F[Refit on all outer-training P4]
    F --> O[Held-out-guide P4 metrics]
    O --> P[P60 support check only after selection]
    P --> G{All promotion gates pass?}
    G -- Yes --> Q[Qualified T01 parent receipt]
    G -- No --> R
```

## Frozen information contract

| Item | Contract |
| --- | --- |
| Parent | One verified `PooledFiniteMeasureBundle` from T00 |
| Count input | CSR raw counts with exact feature and row identities |
| Encoder-fit checkpoint | P4 only |
| Encoder-fit guides | Outer-training guides only |
| Inner selection rows | Guide-stratified P4 holdout inside outer training |
| Protected source rows | Held-out-guide P4 |
| Protected terminal rows | Every P60 cell during selection and refit |
| Candidate dimensions | 8, 16, 32, 48 |
| Update 0/null candidate | Exact global gene-frequency decoder, dimension 0 |
| Dynamics gradients | Disabled |
| Primary loss | Per-count multinomial negative log likelihood, nats/count |
| Primary margin | Learned minus global NLL less than `-1e-4` |
| Uncertainty unit after selection | Perturbation target, not cell or guide |

For each outer fold, ten percent of each training guide's P4 cells, with at
least one cell per guide, forms the inner selection population. The fit and
inner populations are independently capped at 20,000 and 8,192 cells. Row IDs
are canonically sorted before seeded subsampling, and their hashes are written
to `FOLD_INDEX.json`.

The candidate selected in an outer fold is

\[
d^*=\arg\min_{d\in\{8,16,32,48\}} L_d
\]

only among candidates satisfying

\[
L_d-L_0 < -10^{-4}.
\]

If no learned dimension satisfies this inequality, dimension 0 is selected.
This is an ordinary scientific null selection, not a numerical error.

## Mathematical candidates

For a cell count vector \(x\), let \(n=\sum_jx_j\),
\(p_j=x_j/n\), and \(h_j=\sqrt{p_j}\).

The exact global-frequency baseline is fitted from outer-training P4 counts:

\[
f_j=\frac{\sum_i x_{ij}+0.5}{\sum_{i,k}x_{ik}+0.5G},
\]

where \(G\) is the number of features. Its per-cell loss is

\[
L_0(x)=-\frac{1}{n}\sum_jx_j\log f_j.
\]

The `0.5` feature pseudocount gives every retained feature nonzero support.

### V1: uncentered Hellinger PCA

`multinomial_hellinger_pca_v1` fits a low-rank basis \(V_d\) directly to the
P4 Hellinger matrix:

\[
z=hV_d^\top,\qquad
\widehat p=\operatorname{normalize}\!\left((zV_d)^2\right).
\]

The reconstruction is nonnegative after squaring and is renormalized to a
probability vector before multinomial NLL evaluation. This formulation spent
capacity reconstructing the dominant global expression profile, motivating a
separately identified successor rather than a threshold change.

### V2: globally centered residual Hellinger PCA

`multinomial_centered_hellinger_pca_v2` makes the global profile explicit:

\[
c=\sqrt f,\qquad
z=(h-c)V_d^\top,\qquad
\widehat p=\operatorname{normalize}\!\left((c+zV_d)^2\right).
\]

For small matrices the basis comes from a dense SVD of \(h-c\). For large
matrices, a seeded randomized range finder and compressed SVD avoid
materializing the full cell-by-gene matrix. Component signs are canonicalized
by forcing each component's largest-magnitude loading to be positive.

V2 received a new seed, output directory, method identity, implementation
hash, representation ID, and adapter receipt. V1 was never overwritten.

## Conditional post-selection metrics

These metrics are implemented but intentionally run only if every outer fold
selects a learned dimension:

- held-out-guide P4 per-count NLL versus the global decoder;
- held-out-guide pseudobulk gene-composition correlation;
- within-guide P4 split-half latent-centroid distance;
- sister-target versus randomly chosen target-centroid separation;
- split-half distance divided by random-target distance;
- P60 nearest-neighbour support coverage relative to held-out P4 coverage;
- 2,000-draw target-clustered percentile intervals; and
- 20 genuine small count-native refits after guide-target permutation.

P60 support uses a nearest-neighbour threshold calibrated as the 99th
percentile of training-P4 calibration distances. P60 is encoded without
refitting, and terminal coverage must be at least 80% of held-out-P4 coverage.

The complete promotion rule requires all of the following:

1. every fold selects a learned dimension;
2. the target-bootstrap upper bound for NLL improvement is below `-1e-4`;
3. the lower bound for normalized target-shared activity exceeds 0.05;
4. the median split-half/random-target distance ratio is at most 0.5;
5. terminal/source support coverage is at least 0.8 in every fold;
6. target-permutation false selection is at most 0.05; and
7. all previously qualified protected metrics pass.

Because both real candidates stopped at item 1, the downstream metrics were
not evaluated. Their empty Parquet tables, empty encoder archive, zero-valued
bootstrap placeholder, and `not_run_no_selected_representation` null record
are explicit fail-closed artifacts. In particular, the recorded point delta
and interval `[0, 0]` do **not** mean that a learned model tied the null.

`protected_metrics_pass=true` in these receipts means that the exact passed
T00 bundle and complete CountStore parent were reverified and unchanged. T01
did not define a separate continuous protected-metric vector beyond those
parent invariants; future learned candidates should add one if additional T00
diagnostics become eligible for quantitative degradation testing.

## Real Renz development inputs

The external adapter used the pooled Renz/GSE235325 mouse epidermal screen,
not a human HNSCC cohort and not an independent validation cohort.

| Identity | Value |
| --- | --- |
| T00 pooled-data ID | `7313a9d990900148618edc0fe6a72d5f41e2594641b10d48c278370240964f58` |
| CountStore SHA-256 | `cb7bd724b0adf3b74ec7141d6aa5605566c2b8f3f9b2f3d846482bc7b7efafd4` |
| Feature scope | `common34699`, 34,699 assay-common gene-symbol features |
| Fold-assignment SHA-256 | `fead13d3f85755405ddcdd8962f5cc7eef8db46702b569b0a2fe9ae68fe4121a` |
| Retained guides | 495: 445 targeting and 50 controls |
| Retained cells | 277,200: 110,934 P4 and 166,266 P60 |
| Outer folds | 123, 124, 125, and 123 held-out guides |

The four guide folds were inherited from historically exposed Renz model
development and cover the T00 catalog exactly once. Other guides for the same
target remain in training. T01 is therefore a known-target, guide-held-out
development test, not unseen-target or prospective validation.

## Real results

All values below are P4-only inner-validation NLL deltas in nats/count;
positive is worse than the global decoder.

### V1

| Fold | Global NLL | d=8 | d=16 | d=32 | d=48 | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fold0 | 8.028093 | +0.256407 | +0.227898 | +0.213463 | +0.211521 | 0 |
| fold1 | 8.023952 | +0.248619 | +0.220152 | +0.206245 | +0.203762 | 0 |
| fold2 | 8.025998 | +0.247401 | +0.218622 | +0.204275 | +0.202065 | 0 |
| fold3 | 8.034082 | +0.249706 | +0.220393 | +0.205816 | +0.203401 | 0 |

Mean deltas for dimensions 8, 16, 32, and 48 were respectively +0.250533,
+0.221766, +0.207450, and +0.205187. V1 status is `fail_retired`.

### V2

| Fold | Global NLL | d=8 | d=16 | d=32 | d=48 | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fold0 | 8.024258 | +0.233550 | +0.202708 | +0.185275 | +0.183160 | 0 |
| fold1 | 8.027689 | +0.233103 | +0.200140 | +0.182600 | +0.180231 | 0 |
| fold2 | 8.022773 | +0.237764 | +0.207387 | +0.188488 | +0.186294 | 0 |
| fold3 | 8.029755 | +0.227324 | +0.197358 | +0.180659 | +0.178258 | 0 |

Mean deltas for dimensions 8, 16, 32, and 48 were respectively +0.232935,
+0.201899, +0.184256, and +0.181986. V2 status is `fail_retired`.
V2's deltas are numerically smaller, but V1 and V2 used different frozen
seeds and therefore different sampled fit/inner rows. This is not a paired
claim that V2 improved on V1. The valid conclusion is that both were clearly
worse than their own exact global-frequency null in every fold.

## Software surfaces and artifacts

The reusable implementation is in
[`qualification.py`](../src/credo_count_sde_v4/representation/qualification.py).
The typed identity is
[`CountRepresentationBundle`](../src/credo_count_sde_v4/contracts/models.py),
with generated schema
[`count-representation-bundle.v1.json`](../schemas/count-representation-bundle.v1.json).

Python API:

```python
from credo_count_sde_v4 import api

api.qualify_representation(
    destination,
    pooled_bundle=t00_bundle,
    count_store=count_store,
    outer_folds=fold_table,
)
```

CLI:

```bash
credo-v4 qualify-representation \
  --pooled-bundle T00_pooled_data_contract \
  --count-store counts-common.h5 \
  --outer-folds outer-folds.parquet \
  --output T01_count_native_representation
```

Each immutable T01 directory contains:

| Artifact | Meaning |
| --- | --- |
| `TEST_CONTRACT.json` | Frozen component, baseline, margin, and disabled channels |
| `INPUTS.sha256` | T00, CountStore, and fold identities |
| `CONFIG.yaml` | Dimensions, budgets, gates, and seed |
| `CANDIDATE_METRICS.parquet` | P4-only dimension-selection NLLs |
| `FOLD_INDEX.json` | Selected dimension and fit/inner row hashes |
| `ENCODERS.npz` | Selected refit encoder state; empty when dimension 0 wins |
| `PER_GUIDE_METRICS.parquet` | Conditional held-out-guide metrics |
| `PER_TARGET_METRICS.parquet` | Conditional target-level summaries |
| `SUPPORT_METRICS.parquet` | Conditional held-out P4/P60 support coverage |
| `NULL_CALIBRATION.json` | Conditional genuine null-refit receipt |
| `BOOTSTRAP_RESULTS.json` | Conditional target-clustered intervals |
| `CHANNEL_ACTIVITY.json` | Representation activity and channel isolation |
| `SELECTED_MODEL.json` | Dimension, selection checkpoint, and refit status |
| `TEST_RECEIPT.json` | Final pass or `fail_retired` decision |
| `representation.json` | Content-addressed typed bundle |
| `artifacts.json`, `COMMITTED` | Manifest-last atomic publication evidence |
| `SHA256SUMS` | Complete bundle byte verification |

Full verification rehashes the bundle and CountStore, revalidates the T00
parent, checks every typed artifact reference, rejects terminal-informed
selection, confirms fold coverage, checks selected encoder shapes, and verifies
`SHA256SUMS`.

## Tests and release evidence

Focused tests in
[`test_count_representation.py`](../tests/component/test_count_representation.py)
establish that:

- T01 publishes the complete typed receipt surface;
- source and terminal checkpoints stay distinct;
- dynamics gradients and terminal-informed selection remain disabled;
- all expected folds receive one selected candidate;
- complete verification succeeds against the exact T00 and CountStore; and
- replacing every synthetic P60 count realization leaves P4 candidate metrics
  and dimension selection exactly unchanged.

The dev19 repository validation completed with 113 tests passed, one CUDA
smoke test skipped because CUDA was unavailable in that environment, and
86.40% coverage. Ruff, mypy, generated schemas, documentation links,
repository checksums, reproducible wheel/sdist construction, wheel namespace
audit, and a clean wheel-only lifecycle also passed. T01 itself is a CPU
component test; GPU memory is not a scientific selection criterion.

## Interpretation and next allowed action

The global decoder's advantage means these low-rank reconstructions discarded
too much cell-level gene-composition information to earn a latent-state parent
role. The all-positive deltas are much larger than the frozen `1e-4` margin,
so this is not a close-call uncertainty issue or an insufficient-epochs issue.
Neither candidate has trainable epochs, particles, or dynamics parameters.

The next allowed action is to write and freeze a new T01 protocol with a
scientifically distinct count model and its own identity—for example, a
proper held-out-count NB/multinomial encoder with explicit regularization and
training-only selection. The previous null must remain selectable, and the
new protocol must preserve the same protected-row boundary. T02 must not run
until such a T01 receipt passes.
