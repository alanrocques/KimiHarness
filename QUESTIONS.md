# Open questions

Per spec §22 — surface rather than decide.

1. **Exact Moonshot model id.** `kimi-k2-0711-preview` is wired in as the
   default proposer/target, but Moonshot refreshes their catalog regularly.
   Verify against <https://platform.moonshot.ai> before the first paid run and
   update `DEFAULT_PROPOSER_MODEL` / `DEFAULT_TARGET_MODEL` in
   `src/mini_meta_harness/config.py` (and the matching key in
   `src/mini_meta_harness/cost.py::PRICING`).

2. **Target model choice: Kimi vs. Claude Haiku 4.5.** The code already
   supports either — pass `--target-model claude-haiku-4-5` and set
   `ANTHROPIC_API_KEY`. Using Haiku as the target gives a different portfolio
   framing ("cross-provider harness optimization"); using Kimi keeps the whole
   run single-provider and cheaper. Which story do we want to tell?

3. **Second dataset (AG News).** Spec leaves this open. Adding AG News after
   Symptom2Disease works would demonstrate generality. Cost is roughly
   additive (another ~$15 at 12 iterations). Worth it, or ship one dataset
   well first?

4. **Pricing sanity check.** The numbers in `cost.py::PRICING` are rough
   ballparks — $0.60/M input and $2.50/M output for Kimi K2.6, $1.00 / $5.00
   for Claude Haiku 4.5. Verify against the live pricing pages before a
   $15-budget run. The estimator is only as good as these constants.

5. **Per-iteration cost breakdown.** The `Iteration.cost` field in
   `RunSummary.iterations` is currently zeroed — running totals are kept in
   `cost.json` and per-iteration target token counts are recoverable from
   `eval_trace.jsonl`, but proposer tokens per iteration are only recorded as
   part of the running total. If the visualizer needs per-iteration proposer
   cost, we need to persist it (e.g. in a new `proposer_cost.json` per
   iteration dir, under a schema_version bump).
