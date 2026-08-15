# Architecture

Status: implemented engineering architecture for `4.0.0.dev17`.

The package is a sibling distribution. It owns the count-native numerical
recipe and never writes into the frozen CREDO checkout. Numerical modules can
be imported without CREDO; every supported lifecycle API and CLI command first
verifies the exact frozen checkout and vendored source archive.

```text
strict contracts → sparse CountStore → source-only preparation
→ content-addressed compilation → update-based training/checkpoints
→ inference bundle → separate one-shot evaluation → sealed aggregate
```

The three intents are monotone. `count_state` exposes state transport only.
`count_measure` adds a complete-denominator relative-fitness model and exact
Dirichlet–multinomial count objective. `count_context` adds separate
state-context and fitness-context channels plus four same-start contrasts.

The public loader is `credo-v4 open-run`. Development builds do not register a
`credo.recipes` entry point. Stable discovery is reserved for the independently
qualified `4.0.0` release.

No cohort adapter, biological claim, filesystem location, or real input is
embedded in this repository.

The optional dev17 terminal-anchor pilot separates three nested families:

```text
M0 = global training-terminal centroid
M1 = M0 + bounded, analytically fitted sister-guide target shrinkage
M2 = M1 + low-rank source × target interaction
```

M1 uses leave-one-guide-out sister-target predictions on state-fit rows and is
frozen before M2 optimization. The
source is centered and ridge-whitened with `(C + lambda I)^-1/2`; target
interaction embeddings are normalized to remove a scaling gauge. Every
checkpoint is scored both factual and interaction-off. A verified row-level
calibration receipt binds at least 59 genuine null fits, their exact seeds,
implementation, representation, split, support structure, optimizer settings,
and calibrated checkpoint grid. Separate calibrated margins guard M0→M1 and
M1→M2; the outer gate requires M2 to beat both M1 and M0. Update 0 remains the deployable null, controls
and reference mode have exact-zero target effects, and transactional refit uses
all outer-training series after selection.
