# Open questions

Per spec §22 — surface rather than decide.

1. ~~**Exact Moonshot model id.**~~ Resolved: `kimi-k2.6` is the current
   flagship (verified against
   <https://platform.kimi.ai/docs/pricing/chat-k26>, April 2026). Pricing
   baked in: $0.95/M input (cache miss), $4.00/M output. Re-verify before
   any paid run — Moonshot refreshes the catalog regularly.

2. **Target model choice: Kimi vs. Claude Haiku 4.5.** The code already
   supports either — pass `--target-model claude-haiku-4-5` and set
   `ANTHROPIC_API_KEY`. Using Haiku as the target gives a different portfolio
   framing ("cross-provider harness optimization"); using Kimi keeps the whole
   run single-provider and cheaper. Which story do we want to tell?

3. **Second dataset (AG News).** Spec leaves this open. Adding AG News after
   Symptom2Disease works would demonstrate generality. Cost is roughly
   additive (another ~$15 at 12 iterations). Worth it, or ship one dataset
   well first?

4. **Pricing sanity check.** Kimi K2.6 input is billed at two rates:
   $0.16/M on cache hit vs. $0.95/M on cache miss. The estimator assumes
   all input is cache-miss (conservative). If the Anthropic SDK starts
   surfacing `usage.cache_read_input_tokens` against Moonshot's endpoint,
   split `cost.py::PRICING` into hit/miss keys. Claude Haiku 4.5 pricing
   ($1.00 / $5.00) is a ballpark — verify before cross-provider runs.

5. **Per-iteration cost breakdown.** The `Iteration.cost` field in
   `RunSummary.iterations` is currently zeroed — running totals are kept in
   `cost.json` and per-iteration target token counts are recoverable from
   `eval_trace.jsonl`, but proposer tokens per iteration are only recorded as
   part of the running total. If the visualizer needs per-iteration proposer
   cost, we need to persist it (e.g. in a new `proposer_cost.json` per
   iteration dir, under a schema_version bump).
