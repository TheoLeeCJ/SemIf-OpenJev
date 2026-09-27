# Method

## Question and systems

Phase 1 tests whether open, generation-free readouts reproduce the useful part of Jev's public claim: accept unstructured state plus runtime-defined natural-language decisions and return typed scores cheaply enough to embed in ordinary software.

The direct system prompts frozen Qwen3.5-4B with a state, criterion, and 2-16 described options. It performs one native forward pass and applies a softmax only to the logits of fixed uppercase answer tokens. It does not decode a token.

The reranker system follows Qwen3-Reranker-4B's native yes/no contract. Each candidate answer becomes a separate query/document relevance proposition. The system computes `logit(yes) - logit(no)` for every option and softmaxes those log-odds across options. That last normalization is our comparison rule; it is not part of the upstream reranker's calibration contract.

## Frozen evaluation matrix

Prompts, IDs, labels, task semantics, revisions, and metrics were frozen before the full reranker outputs were evaluated. The complete local matrix had 706 rows:

| Source | Rows | Purpose |
|---|---:|---|
| Project-authored | 144 | Evidence interpretation, rule application, candidate selection; original and missing-evidence cases |
| WANLI | 256 | External natural-language inference |
| TypeSafe public evaluation subset | 102 | Distribution/reference agreement on 20 available cases |
| Every public lab artifacts | 204 | Judgment grid, retrieval, company knowledge, composed action policy |

Unlike tasks were not collapsed into one accuracy number. Hard-label tasks use full-denominator accuracy, balanced accuracy, macro F1, NLL/Brier where applicable, and source-group bootstrap intervals. Retrieval reports query ranking metrics. TypeSafe rows compare distributions and use an equal-case macro so cases with more questions do not dominate.

## Optional answer-set cohorts

Synthetic FAQ material may declare more than one semantically acceptable option. These records are an optional answer-set cohort, separate from the frozen 706-row phase-1 matrix and its headline claims.

- A legacy hard-label record continues to use its integer `label` option index unchanged.
- An answer-set record provides a nonempty, unique `answer_set` of declared semantic option IDs and a required `target_distribution`. Target and predicted distribution values must be finite in `[0, 1]` and sum within `1e-4` of one. The IDs with strictly positive target mass must equal `answer_set` exactly; accepted target vectors are renormalized to unit mass before expected metrics are calculated.
- A multi-answer record omits `label`, `target_option_id`, and `target_letter`. A singleton answer-set record may omit that projection, or supply all three fields. When supplied, the integer label, semantic ID, and `A`–`P` letter must describe the same declared option and singleton answer set.

Answer-set reports provide expected NLL under the normalized target distribution, expected multiclass Brier score, probability mass assigned to the acceptable set, and top-1-in-set. Exact top-probability ties count as in-set only when every tied option is acceptable, so reordering options cannot change that metric. `top_1_in_set` uses every answer-set row as its denominator, with missing or invalid outputs counted as failures; the `*_valid_decisions_only` companion uses only parsed decisions. Expected NLL, Brier, and acceptable-set mass are reported for the full cohort only when every row has a valid distribution, with `*_valid_distributions_only` variants otherwise.

Native predictions provide no probability metrics and never receive a fabricated one-hot distribution; they contribute only top-1-in-set. A native prediction for a complete singleton hard projection can still contribute its hard-label classification result, while its NLL, Brier, and probability-mass fields remain unavailable. Hard-label accuracy and its screening and paired-comparison paths accept legacy hard rows and fully validated singleton projections only; unprojected singleton and all multi-answer rows remain outside those paths.

No headline benchmark claim changes through this optional cohort. Heldout or sourced FAQ material requires independent review before it enters an evaluated population or supports a claim.

The TypeSafe comparison is the 102 public rows that could be aligned locally, not its advertised 711-row aggregate and not a live Jev run. Published Jev, Opus, and Sol values were read from those public records. Raw third-party fixtures are excluded from this candidate.

The exact evaluated IDs are committed in `benchmarks/manifests/source-selection.jsonl`. WANLI uses revision `61c95318fd71c55b6ba355d76253254615f387ec`: malformed or over-4,000-character rows and components touching pilot-training sources were excluded, remaining IDs were sorted then shuffled with seed 291607, and 86 entailment/85 contradiction/85 neutral rows were selected with at most one row per connected premise/pair-ID component. Premise becomes state; hypothesis becomes the criterion; entailment/neutral/contradiction map to supported/insufficient/contradicted.

TypeSafe extraction reads the four hashed `*-cases.js` snapshots listed in `benchmarks/fetch_sources.py`. It keeps published, successfully run Choice/Noul nodes having one unambiguous document/question binding, a released reference answer, and a unique reference-distribution argmax. Selection is outcome-blind round-robin over workflow, case, and primitive, with hashed-ID order inside buckets. Score primitives and tied targets are excluded from the 102-row comparison. Every mappings and original experiment data are available from the directly linked experiment JSON and source archive.

## Perturbations

Thirty-six owned original cases received three output-blind variants: reverse the displayed option order while preserving semantic IDs, wrap the criterion in meaning-preserving wording, and append irrelevant owned context. A separate 36-row missing-evidence population tests whether a system selects `insufficient`. Stability is measured after aligning probabilities by semantic option ID.

## Shape-matched systems benchmark

An owned fixture contains 37 states and 21 fixed binary criteria per state, giving 777 decisions. States are roughly 8,000 characters and exercise repeated-context computation. It matches the count geometry of the public Every/Jev demonstration, but does not reproduce its unpublished documents, token lengths, hardware, API path, or model. Therefore it is a systems measurement, not a Jev head-to-head benchmark.

Direct modes are fresh batch-one scoring, serial suffixes after one state prefill, and parallel suffix branches after one state prefill. The reranker repeats the state for two independent yes/no option pairs per binary decision and tests ordinary pair batching. Timings use one RTX 3090 with a warm-loaded BF16 model and include prompt construction, tokenization, transfers, forward passes, and CPU readout; model loading and result-file writes are outside the timed region.

## Interpretation rules

- A forced typed output can still be semantically wrong.
- Softmax over allowed tokens is conditional on the supplied alternatives; it is not calibrated operational confidence.
- Prefix-cache speedups are implementation results, not evidence about Jev's disclosed architecture.
- A reranker is expected to be strongest on ranking. Its categorical threshold metrics should not be confused with ranking quality.
