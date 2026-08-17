# Dev33-B G00C extraction canary

Last verified: 2026-08-17

Status: passed non-promotable engineering canary; claim-bearing G00C not run

Dev33-B executed the one action authorized after Dev33-A: a disposable,
CPU-only extraction and sampler canary against accepted G00A-v2/G00B-v2. The
authoritative external attempt is
`../credo_v4_gse314342_g00_20260816/G00C_EXTRACTION_CANARY_DEV33_B_A3`.
Its detailed `RUN_REPORT.md`, independent receipt, post-publication audit, and
`FINAL_CANARY_ARTIFACTS.sha256` are the execution authorities.

## Frozen information set

- LODO fold 0: held-out D1; ordered training donors D2, D3, D4.
- Payload: 50,000 training-fit + 4,096 training-validation + 4,096 held-out
  Rest source-query rows = 58,192 rows.
- Held-out D1 Stim8hr/Stim48hr expression: unopened; metadata roles only.
- Primary surface: fixed 256-feature training-only ranking.
- `CUSTOM001_PuroR`: separate one-column technical sidecar.
- Released Gold/Silver/Null tiers: unused and forbidden.
- Contract ID: `43091c1f2f81f326cabfa62c860162d13331eed6af9a696945e84376a9c41a5c`.

## Results

- Both writer paths produced byte-identical 55,500,216-byte primary HDF5
  files and byte-identical 5,166,776-byte PuroR sidecars.
- The resumed writer stopped after one committed 4,096-row block and published
  no final payload before resuming.
- Bounded source verification checked 58,192 rows, 13,043,890 nonzeros, and
  148,360,514 total counts in 15 blocks. The largest loaded block contained
  967,036 nonzeros.
- The 32-microbatch sampler replay matched exactly after interruption at
  cursor 12 for row IDs, weights, thinning draws, RNG states, cursor, and
  balance summaries.
- Maximum external process RSS was 3,531,415,552 bytes, below the frozen
  16-GiB ceiling.
- Protected held-out stimulated expression reads were exactly zero.

The independent status is `pass_engineering_canary`; `promotion_eligible`,
`may_parent_g00d`, and `may_parent_g04` are all false. No model, GPU,
performance metric, or biological statistic was produced.

## Preserved failures

A1 is retained as a failed duplicate-reader orchestration attempt. A2 is
retained as a failed monitor-contract attempt after a zero-missing-sample rule
proved brittle. Neither published compact evidence nor read protected held-out
stimulated expression. A3 preregistered its bounded monitor tolerance before
expression access and did not reuse A2's feature artifact.

## Stop rule

This canary does not launch the full G00C curve. Before a claim-bearing fold,
freeze the feature margin, cell-count epsilon, serial dependency, 59 draw
seeds, replay subset, row hashes, maximum grid, and final publication rule.
G00D and G04/G07/G08 remain blocked.
