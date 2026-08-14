# Frozen CREDO compatibility

Supported workflows require both:

- checkout commit `6f4f57c114606476cb6e93c6d52dc34628173a4d` with a clean worktree;
- archive SHA-256
  `2020b0f2549bb98b2b0fbb9384783d39be3ce11aeef9ae2bfecf232ae07fc684`.

The archive is vendored under `vendor/`; its checksum is in
`vendor/SHA256SUMS`. The package never installs into the `credo` namespace.
Numerical unit tests import without CREDO. Supported API/CLI entry points use
the read-only compatibility verifier before performing work.

For a wheel installed outside this workspace, set `CREDO_V4_WORKSPACE_ROOT` to
a directory containing `CREDO/` and either `vendor/credo-6f4f57c.tar` or the
sibling V4 repository. The verifier never repairs, updates, or cleans CREDO.
