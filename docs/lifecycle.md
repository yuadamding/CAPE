# Lifecycle

The append-only state machine is:

```text
RESOLVED → PREPARED → COMPILED → TRAINED → FINALIZED → EVALUATED → SEALED
```

Typical execution:

```bash
credo-v4 synthetic --output /tmp/credo-v4 --intent count_context
credo-v4 run-all /tmp/credo-v4/config.yaml --device cpu
credo-v4 verify /tmp/credo-v4/work/sealed --level full
```

Preparation fits only `InformationSet.fit_rows`; protected rows may be encoded
after freezing but never influence encoder identity. Compilation validates and
hash-binds; it does not fit. Training publishes immutable checkpoint
generations. `resume` requires the identical compiled contract. `fork` starts a
new attempt from compatible model weights and a new compiled protocol.

Evaluation is one-shot: checkpoint selection and eligibility are frozen first.
The evaluator publishes a separate audit and cannot mutate the inference
bundle. Sealing creates a read-only aggregate.

Verification levels are manifest, content, reload, resume, and full. Full
verification rehashes the bundle and count store, reconstructs the model, and
runs a bounded prediction.
