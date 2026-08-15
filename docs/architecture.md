# Architecture

Status: implemented engineering architecture for `4.0.0.dev15`.

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

The optional dev15 terminal-anchor pilot separates shared reference continuation
from factual-only sister-guide target and source × target terms. The source is
centered and ridge-whitened on state-fit rows; the interaction is rank limited,
zero initialized, exactly zero for controls, and removed in reference mode.
Update 0 is persisted as the deployable null; training-only selection chooses
among the null, target-only, and target-plus-interaction families under a
positive margin, then refits the chosen family on all outer-training series.
