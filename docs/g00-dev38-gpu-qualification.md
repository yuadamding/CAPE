# G00C Dev38 GPU qualification

Last verified: 2026-08-21 (America/Chicago)

Status: **engineering-qualified; no Dev37 promotion or biological claim**

## Decision

G00C should use three explicitly separated execution envelopes:

1. The expanded-row envelope preserves the Dev37 PCG64DXSM variate order and
   uses CUDA only for associative integer accumulation. A3 demonstrated exact
   CPU/CUDA equality on its preregistered synthetic surface.
2. The compact CPU-thinning envelope collapses repeated rows before drawing
   one binomial variate. It preserves the binomial law and has exact CPU/CUDA
   reduction equality, but changes PCG64DXSM variate consumption and therefore
   requires a new authority.
3. The high-throughput envelope keeps that compact count surface resident on
   the GPU and uses CUDA-native Philox binomial draws. It is much faster, but
   also requires a new preregistered authority before scientific use.

No H100 canary accessed cohort data, trained a model, or supports a biological
claim.

## CPU bottleneck and behavior-preserving correction

The original sampler rebuilt its hierarchy for all 590 plan entries. Each
entry scanned as many as 16,924,672 hierarchy rows, rebuilt target and guide
arrays inside the draw loop, performed a Pandas scalar lookup per draw, and
appended one Python dictionary per trace row.

`g00c_sampler_v3.py` now:

- constructs five reusable prefix indices (50k, 100k, 250k, 500k, and 1M);
- reuses the 1M index across every feature candidate and draw;
- precomputes source, target, guide, and row lookup arrays;
- preallocates typed NumPy output vectors; and
- retains the original PCG64DXSM call order, state hashes, weights, trace
  schema, and restart behavior.

On the real D1 authority hierarchy, ten 32,768-draw entries completed in
18.6614 seconds, or 17,559 draws/second. At that measured rate, four complete
19,333,120-row replays project to about 1.22 hours. The previous canary-based
estimate was approximately 34--60 hours.

## CUDA implementation

`g00c_refit_cuda.py` adds three intentionally distinct paths.

### Expanded Dev37-compatible path

The expanded path retains every sampled row in its historical order, performs
the frozen PCG64DXSM binomial thinning on CPU, and transfers only integer
contributions and flattened checkpoint-feature indices. CUDA uses int64
`index_add_`. Integer addition is associative, so atomic scheduling order
cannot change the integer result. A hard allocator ceiling defaults to 30 GiB
and must remain below physical device memory.

The A3 H100 canary covered 16,384 expanded rows and 2,097,152 nonzero count
entries. It proved:

- exact equality with the Dev37 `_thin_and_accumulate` CPU reference;
- an identical PCG64DXSM thinning digest;
- H100 80GB HBM3, compute capability 9.0;
- PyTorch 2.6.0+cu124 and CUDA 12.4; and
- 8,486,912 peak allocated bytes, below the 30-GiB ceiling.

This is behavior-compatibility evidence, not an amendment to the sealed Dev37
implementation binding. Production use still requires authority-bound code,
full-surface replay, and the existing fail-closed evidence checks.

### Compact CPU-thinning path

The compact path replaces repeated sampled rows with row multiplicities and
draws from `Binomial(count * multiplicity, 0.5)`. CPU and CUDA use the same
PCG64DXSM draws, and CUDA only performs int64 accumulation, so they are exactly
equal to each other. They are not byte-comparable to Dev37 because one compact
variate replaces multiple historical variates.

This path is not faster for one isolated reduction. A3 measured 1.5496 seconds
for the compact CPU reference and 2.4119 seconds for the cold compact CUDA
call. CPU thinning, transfer, and CUDA initialization dominate this surface.

### GPU-resident path

The resident path uploads flattened indices, effective count trials, weights,
and output storage once, then performs CUDA-native binomial thinning and int64
reduction for many seeds. A3 measured:

| Metric | Result |
| --- | ---: |
| Synthetic sparse matrix nonzeros | 8,388,608 |
| Sampled rows | 1,000,000 |
| Unique rows | 65,536 |
| Timed reductions | 10 |
| Median seconds per reduction | 0.00567212 |
| Speedup over CPU reference | 273.20x |
| Mean relative error to binomial expectation | 2.51e-7 |
| Same-seed restart equality | exact |

This result is an engineering performance canary. CUDA uses Philox rather
than the frozen PCG64DXSM stream, so the output is not byte-comparable to a
Dev37 refit. The implementation states this boundary in its public docstring
and reports `authority_compatible_with_dev37=false`.

## Kubernetes evidence

All three attempts used one H100 Job in context
`yding4_yn-gpu-workload@kubernetes-admin@kubernetes`, namespace
`yn-gpu-workload`, and the pinned image:

`hpcharbor.mdanderson.edu/yding41/credo_env@sha256:138d7191792ce1139708751f109817a976c5b96ba3a00be3dff88173b62eb21a`

Durable roots:

- `/rsrch8/home/bcb/yding4/perturbseq/runs/credo-v4-g00c-dev38-gpu-20260820-a1`
- `/rsrch8/home/bcb/yding4/perturbseq/runs/credo-v4-g00c-dev38-gpu-20260820-a2`
- `/rsrch8/home/bcb/yding4/perturbseq/runs/credo-v4-g00c-dev38-gpu-20260820-a3`

### Attempt results

| Measurement | A1 | A2 | A3 |
| --- | ---: | ---: | ---: |
| Compact CPU reference, seconds | 1.737925 | 1.436526 | 1.549638 |
| Cold compact CUDA, seconds | 2.087996 | 1.937974 | 2.411943 |
| Compact CPU/CUDA equality | exact | exact | exact |
| Resident CUDA median, seconds | not tested | 0.00568604 | 0.00567212 |
| Resident speedup over CPU | not tested | 252.64x | 273.20x |
| Resident same-seed restart | not tested | exact | exact |
| Resident mean relative error | not tested | 2.51e-7 | 2.51e-7 |
| Expanded Dev37-order equality | not tested | not tested | exact |

A1 preserved the negative cold-start performance result. A2 established the
resident performance result. A3 added the expanded Dev37-reference comparison
and independently repeated the resident measurement. Each root contains immutable
publication, admission, suspended binding, activation, terminal, worker-log,
result-verification, and cleanup evidence. All exact Jobs and their UID-owned
Pods were deleted with UID/resourceVersion preconditions and verified absent
twice.

### Immutable identities

| Attempt | Object | SHA-256 or UID |
| --- | --- | --- |
| A1 | Source archive | `4261919a82bdeb482fbebf6cd010745d54ab2ce88842f78d12fd3e183acaf82f` |
| A1 | Job UID | `0ca6a0e2-a737-4c2b-90eb-2dc704b07a8c` |
| A1 | Result | `e438090cec77a98f13bc05b92955ace8c30b11a27686da4127ca57a88217824b` |
| A1 | Result verification | `8526334962dbcf74eebff40580e36eafa8eabd62738836cf6ac5c322637faf4e` |
| A1 | Cleanup receipt | `603ba19717c0a06ca2144c7a3f7a5c221f66973686aa81fbdaf3503d5f1e7be0` |
| A2 | Source archive | `1aefff6476726ab0b5e832b6602fb96db70ca7ad6640c43650782d8aa9f8c5a6` |
| A2 | Job UID | `0b133c71-cda6-44b9-a703-734d6dfbd54c` |
| A2 | Result | `0b270d6ef24755938bd7bc76f0588e18951ae6cba496be36c969c88a405ea356` |
| A2 | Result verification | `ee8465cea412c6261b1d1abd6437e70cc098f74adc0df97225082cd65b8bce2c` |
| A2 | Cleanup receipt | `5348f4a634cfcf70156d937b8096b2300c7ebc5df7d07372aa15d1ec9e73884f` |
| A3 | Source archive | `d2cffd4805cb7fe710affbaba919c442506d34171e5d49d249dc4633eb8d169b` |
| A3 | Job UID | `95b37305-5f59-43b2-bd01-355dd4f5d800` |
| A3 | Result | `958b442e5dbec3aac4cb7e092c9c2c4d2ab817cd4c0325e64e9aeeb49e4bdb31` |
| A3 | Result verification | `d814eaf1086c96ab9281466959e3264957a5961f44563dc17474ecce7ac60511` |
| A3 | Cleanup receipt | `97ae4521b98905c0715cf4105cca39cafdfe792a96a3e6f9f59520aa8078c95a` |

Admission increased the requested CPU and host memory to 24 CPUs and 200 GiB
and injected an A100 node selector while retaining the required H100 product
affinity. Runtime startup proved that all three workers received an H100. Future
manifests must continue auditing the admitted object rather than trusting the
rendered resource values.

## Scientific promotion requirements

Before either compact path can replace Dev37 sampling or thinning, a new
authority must freeze and test all of the following:

1. CUDA Philox key/counter derivation for every plan entry, draw, row, and
   feature, including restart cursor behavior.
2. H100 image, Torch/CUDA versions, precision, allocator ceiling, device
   capability, and deterministic-kernel envelope.
3. Positive, null, adversarial, and ablation tests for distributional and
   decision-level equivalence across the complete 59-draw grid.
4. Feature-width and cell-budget selections under both the Dev37 reference
   and proposed stream, with a preregistered tolerance and identical terminal
   decision requirement.
5. Streaming or sufficient-statistic evidence that avoids retaining four
   complete 19.3-million-row traces in memory while preserving independent
   replay and support auditing.

Until those gates pass, the safe operational choice is the optimized historical
sampler plus the expanded-row CUDA accumulator only where end-to-end profiling
shows a benefit. Cold single reductions should remain on CPU. G00D and
biological interpretation remain blocked by G00C.

## Verification

Focused local validation before the final broad run:

- 8 passed, 3 CUDA-only skips, 30 deselected;
- Ruff passed;
- mypy passed for the changed sampler/CUDA modules.

An earlier broad run on `/mnt/seagate` produced 291 passes and three failures
only because that filesystem forbids symlink creation. The final post-A3 run
using `/dev/shm` completed with 294 passes, four expected local CUDA skips, and
one pre-existing near-constant-input warning in 74.67 seconds. Full Ruff, mypy
over 76 source files, documentation-link, and `git diff --check` validation all
passed. The H100 Kubernetes canaries supplied the CUDA runtime evidence.
