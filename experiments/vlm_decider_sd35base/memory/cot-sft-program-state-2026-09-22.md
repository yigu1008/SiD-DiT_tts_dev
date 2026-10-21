---
name: cot-sft-program-state-2026-09-22
description: Where the strong-to-weak SFT program stood when the user paused it on 2026-09-22 (all arms negative; what remains; where records live)
metadata: 
  node_type: memory
  type: project
  originSessionId: bafccd3a-99c6-4610-8c23-ab7f781db088
  modified: 2026-09-22T00:01:12.671Z
---

As of 2026-09-22 the strong→weak SFT program in ~/project/CoT_SFT_dev is paused with nothing running.
Every arm measured (rows A–V in DIAGNOSIS.md, one VERIFY.md section each) is at or below base for
post-trained Qwen3 thinking students (0.6B, 1.7B, 4B): R1 traces, same-family 8B traces, self-distillation,
mixtures, zone selection, trace edits, and four token-loss weightings. Mechanism: token weighting sets the
closing rate (44→97% across masks) but not correctness; the only positive was Qwen3-1.7B-Base (no thinking prior)
with budget. Eval prompt suffix mismatch understates the regression (~2.5 pts), protocol kept.

**Why:** the user asked to pause and move to something else; the next session should not re-derive this.

**How to apply:** read DIAGNOSIS.md (top table + "Is it just that the model is too weak?") before proposing
new arms. Runner scripts are in scripts/runners/ (copied from the session scratchpad). Open offer not taken:
a positive control (Qwen3-0.6B-Base on s1K) for the write-up. Related: [[s2w-standing-preferences]].
