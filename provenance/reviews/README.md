# External review provenance

Status: portable hash authority for review records received outside this repository.

`review-hashes.v1.json` preserves the SHA-256 identity of each historical review
used by the development receipt. The original review bodies were received as
conversation attachments and are not runtime inputs. Receipt regeneration reads
this committed authority; it never accesses a user-specific attachment path.

The Dev34-A independent review is bound as
`gse314342_g00_dev34a_independent_review_sha256`. It authorizes only the
expression-free Dev35 execution-authority work. It does not authorize G00C
expression access, training, GPU execution, G00D, or a biological claim.
