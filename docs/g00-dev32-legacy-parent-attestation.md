# G00 Dev32 non-retroactive legacy-parent attestation

Last verified: 2026-08-17
Status: contract and verifier implemented; real wrapper execution remains an
external L1 operation
Authority: legacy-parent contracts and verifier in `4.0.0.dev32`

## Decision

Dev31-B0 A1 correctly stopped before source access because the accepted Dev29
directory was published under a historical `SHA256SUMS` boundary and has no
original `artifacts.json` or `COMMITTED` marker. Dev29 remains accepted under
that historical boundary. Dev32 neither mutates Dev29 nor relaxes the native
manifest-last parent type.

Dev32 adds a second, discriminated parent type:

```text
NativeManifestLastBoundary
    | LegacyChecksumAttestedBoundary
```

The legacy type is admissible only with a separately committed sibling wrapper
and its passed receipt. There is no optional-file or automatic fallback path.

## Wrapper assertions

The wrapper may prove current checksum syntax, complete regular-file coverage,
exact file bytes, accepted G00A/G00B identifiers and relationship, row-universe
and guide/target catalog identities, and that the wrapper process wrote no
parent file and opened no raw source matrix.

It must explicitly retain these historical statements as false:

- original artifacts manifest present;
- original committed marker present;
- original manifest-last publication proven;
- original atomic publication proven; and
- original transactional publication proven.

The evidence role is literally
`nonretroactive_legacy_checksum_parent_attestation`.

## Fail-closed verification

The builder and independent verifier reject malformed or noncanonical checksum
lines, duplicates, traversal, missing or mismatched files, uncovered regular
files, symlinks, special files, wrong parent identifiers, a wrapper inside the
parent, false original-atomicity claims, a failed or unrelated wrapper receipt,
and attempts to route a native parent through legacy mode.

The wrapper is published manifest-last with its own `artifacts.json`,
`COMMITTED`, and `SHA256SUMS`. Its source directory is an exact sibling of the
unchanged historical parent.

## Scientific boundary

This release performs no raw GSE314342 source scan, feature selection, compact
extraction, G00C, G00D, GPU, model, or biological operation. A passed external
L1 wrapper must be reviewed before a fresh B0-A2 attempt may open source
matrices for numerical authority.
