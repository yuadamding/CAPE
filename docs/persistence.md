# Persistence and resume

Model, optimizer tensor leaves, and RNG tensors use a deterministic documented
safetensors wire format. Typed structure uses canonical JSON; tables use
Parquet; large sparse or dense arrays use HDF5. General pickle is forbidden.

Publication is manifest-last on one filesystem:

1. write into a unique temporary directory;
2. close and fsync payloads;
3. write a path-sorted checksum manifest;
4. reopen and verify every declared file;
5. write `COMMITTED` last;
6. fsync and atomically rename;
7. fsync the parent;
8. atomically publish and fsync any convenience pointer.

Committed destinations are never overwritten. Incomplete temporary generations
are ignored and can only be moved with the exact orphan-quarantine operation.
Extra, missing, truncated, corrupt, symlinked, overlapping, or path-escaping
content fails verification.

Training randomness is counter-derived from update and seed. Checkpoints occur
at optimizer boundaries and bind model, named optimizer tree, RNG, sampler,
training state, compiled ID, and parent checkpoint. Deterministic CPU tests
prove interrupted and uninterrupted training produce identical final weights.
