# Dev32 B0-A2 source-plane execution record

Last verified: 2026-08-17

Status: accepted GSE314342 source-plane qualification; no model or biological result

The authoritative data remain outside Git at
`/home/yding1995/opscc_sc/credo_v4_gse314342_g00_20260816`. This page and the
[lightweight provenance index](../provenance/g00/dev32-b0-a2/PROVENANCE_INDEX.json)
make the accepted execution discoverable without duplicating its 318 MB locator
or the 1.736 TB source matrices.

## Accepted identities

| Object | Identity |
|---|---|
| source-plane amendment | `628e23e017b7384fae8af2f50c38047255905bb166a1dfdc9c10b4e0dc6e8750` |
| amendment receipt | `fe5431cdd37a4012a5a6e66333ffbad31190180cd24820e2211e3bbc14c70ca2` |
| G00A-v2 | `b12978ea37328779a919320901ee6b71d2a374e6de89ea1d1e74aef5dd6103b9` |
| G00B-v2 | `ee4dc94c06d0eff507042f6571241c63c7d916e6055f84154fed0efa395d423b` |
| decision receipt | `1f0ff03689a4315837c3305d2da12f99e059b1129cde5c9d9d255e40ded5560b` |
| independent verification | `05ca0efad4744fa044632a01737bc1bcf8194deb3e8b4825b1ed96ca321b5880` |

The run rehashed and numerically scanned all 12 source matrices: 1.735835 TB,
21,996,842 eligible rows, 90,997,745,441 eligible nonzeros, 18,130 features,
and maximum observed count 10,569. The accepted catalog has 25,956 guides
(24,972 targeting and 984 controls) and 12,732 target/control categories.

## Eligibility interpretation

The sealed predicate remains exactly:

```text
guide_group == targeting single sgRNA AND low_quality == false
```

Here, `targeting single sgRNA` is the literal raw categorical value for an
accepted single-guide assignment class. It includes both targeting and control
guide identities. Biological targeting/control identity comes from the
accepted crosswalk, not from interpreting that raw category name. The
[append-only interpretation](../provenance/g00/dev32-b0-a2/ELIGIBILITY_INTERPRETATION_AMENDMENT.json)
records 24,972 targeting and 984 control guides without changing the eligible
row hash or sealed B0 bundle.

## Failure dispositions

The initial v2 publication failed before `COMMITTED` because an unused H5AD
category, `multi_sgRNA`, was absent from the eligible guide catalog. A hashed
finalizer resolved only categorical codes observed among eligible rows and did
not reopen raw matrices. The failed publication remains permanently
ineligible; its compact disposition and the exact execution amendment are
retained under [`provenance/g00/dev32-b0-a2`](../provenance/g00/dev32-b0-a2/).

The earlier L1-A1 wrapper is also permanently ineligible because it recorded
the wrong builder commit. L1-A2 is its accepted successor.

## Scope

B0-A2 establishes source-byte, CSR-numerical, categorical, provenance, and
publication authority only. G00C is parent-eligible but was not run. G00D,
G04, G07, G08, model performance, and biological interpretation remain
blocked. The exact external checksum index is mirrored as
[`EXTERNAL_ROOT_SHA256SUMS`](../provenance/g00/dev32-b0-a2/EXTERNAL_ROOT_SHA256SUMS).
