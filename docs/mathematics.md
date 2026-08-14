# Mathematical contract

Particles represent a normalized state law and an optional relative mass. The
drift is a shared reference component plus an exact-zero-for-control target
residual. Diffusion is shared reference dispersion and requires an explicit
passing inner-validation ablation. State selection is globally centered over
each complete series law, so it changes composition but not total mass.

Measure intents use target-shared scalar fitness. Fitness is centered across
every contributor in a physical pool using source exposure weights. There is
no M2 state-dependent mass residual and no free endpoint-fitted guide-capture
nuisance.

Every count event evaluates a complete physical-pool denominator, including
registered zeros. Source exposure is smoothed by the frozen constant `0.5`.
Probabilities use one full softmax. Concentration is
`softplus(log_concentration)` exactly once. The Dirichlet–multinomial uses
float64 `lgamma` and is tested through totals of at least `1e8`.

State and exact-count updates are separate sampler events. A count event's
graph contains only scalar-fitness parameters; it cannot reopen drift or
diffusion.

For the development terminal-anchor family, the source-conditioned successor
uses

```text
z_terminal = global_training_terminal
             + scale * tanh(W_out tanh(W_hidden z_source + b_hidden)).
```

`W_out` is initialized to exact zero, so this model contains the global
training-terminal predictor as a literal nested null. With
`source_carryover_alpha = 0`, every integration step evaluates the same
original-source-conditioned anchor; it does not accidentally recurse on its
own previous prediction. Target-anchor offsets and the global anchor can be
frozen independently. When a gene decoder is trained in the same update,
state gradients are enabled only by the explicit
`train_state_with_gene_decoder` contract. Scientific checkpoint selection uses
the configured training-only state-validation split, not decoder loss or the
outer fold.

Default integration uses a physical grid:

```text
steps = ceil(duration / maximum_step)
step_size = duration / steps
```

Decoded gene output is a composition. It never borrows protected endpoint
library depth.
