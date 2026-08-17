# Dev33-A G00C contract hardening

Last verified: 2026-08-17

Status: implementation and adversarial-test boundary; extraction canary now
passed separately under the [Dev33-B record](g00-dev33b-extraction-canary.md)

Dev33 is additive. The accepted Dev32 contracts and B0-A2 artifacts remain
immutable. New claim-bearing G00C work must use the v3 fold contract and v2
execution/decision evidence.

## Hardening decisions

1. **Sample-size extension is fail-closed.** On the base grid, a candidate
   below one million cells must qualify. If only the one-million reference
   qualifies, the result is `extension_required`, selects no rows, and cannot
   pass G00C. A separate extension contract must bind that stop receipt and add
   exactly two million cells.
2. **Row set and row order are distinct.** The execution bundle binds both the
   sorted row-set hash and the byte-exact ordered-row hash. The verifier derives
   `donor_id, physical_time_hours, checkpoint, target_id, guide_id, source_row,
   row_id` order from G00B and requires contiguous donor/checkpoint/target/guide
   runs.
3. **Compact verification is streaming.** Only a bounded `indptr` window and
   its matching index/value slice are loaded for each block. Count arithmetic
   casts out of `uint16`. The receipt binds block size, maximum loaded
   nonzeros, total rows/nonzeros/count sum, peak process RSS, source-handle
   bound, and physical-block count.
4. **Refit summaries are insufficient.** Every row now binds draw, seed, fit
   and validation row hashes, thinning, configuration, initial/final states,
   validation total, NLL sum, NLL/count, and status. Validation requires an
   executable replay implementation. It replays all selected-candidate and
   reference-candidate fits plus a preregistered subset of every other
   candidate; a summary-only table or absent replay executor fails.

Committed schemas are `fold-native-compact-view.v3.json`,
`g00c-execution-bundle.v2.json`, `g00c-decision-receipt.v2.json`, and
`g00c-compact-verification-receipt.v2.json`.

## Stop rule

This Dev33-A release does not read protected held-out stimulated expression, rank
features, choose a cell budget, materialize G00C, use a GPU, or produce a model
or biological result. The next permissible action after independent review is
the disposable 50,000-row/256-feature one-fold extraction canary. That canary
subsequently passed as Dev33-B engineering evidence, but it cannot be promoted
into the G00C parent.
