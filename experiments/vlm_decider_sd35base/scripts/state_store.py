#!/usr/bin/env python3
"""Decision state for Jev: per-prompt trajectory memory + a cross-prompt wording base.

PromptState  -- what has already been tried ON THIS PROMPT and how it turned out:
                per-wording visit counts and mean terminal reward (the same evidence UCB1 uses),
                the action path to the current node, and the checklist delta each edit produced.
WordingBase  -- what has worked ACROSS prompts: for each (checklist element type, operator) pair,
                how often the edit actually flipped that element to `present` in the child's readout.
                Rendered as a short prior and injected into the proposer and into Jev's state.

Both render to compact text blocks meant to be pasted into a Jev `state` or a VLM instruction.
The base is append-only JSONL so it can be built on a training split and FROZEN for evaluation
(`WordingBase(path, frozen=True)` refuses writes) -- otherwise test numbers are contaminated.
"""
from __future__ import annotations
import json, os, threading
from collections import defaultdict

class PromptState:
    def __init__(self, prompt_id, prompt, wordings):
        self.prompt_id = prompt_id; self.prompt = prompt; self.wordings = wordings
        self.n = defaultdict(int); self.s = defaultdict(float); self.best = defaultdict(lambda: -1e9)
        self.node_status = {}          # node_id -> [{id,status},...]
        self.node_wording = {}         # node_id -> wording index
        self.edits = []                # realised edit outcomes on this prompt

    def credit(self, wording_idxs, score):
        for wi in set(int(w) for w in wording_idxs):
            self.n[wi] += 1; self.s[wi] += float(score); self.best[wi] = max(self.best[wi], float(score))

    def record_status(self, node_id, wording_idx, statuses):
        self.node_status[node_id] = statuses; self.node_wording[node_id] = int(wording_idx)

    def edit_outcome(self, parent_id, child_id, operator):
        """Checklist delta between a parent node and a child reached by changing the wording."""
        p, c = self.node_status.get(parent_id), self.node_status.get(child_id)
        if not p or not c: return None
        pb = {x["id"]: x["status"] for x in p}; cb = {x["id"]: x["status"] for x in c}
        fixed = [i for i in pb if pb[i] in ("missing", "wrong") and cb.get(i) == "present"]
        broke = [i for i in pb if pb[i] == "present" and cb.get(i) in ("missing", "wrong")]
        out = {"parent": parent_id, "child": child_id, "operator": operator, "fixed": fixed, "broke": broke,
               "n_present_before": sum(1 for v in pb.values() if v == "present"), "n_present_after": sum(1 for v in cb.values() if v == "present")}
        self.edits.append(out); return out

    def render(self, cur_wi, path_actions, max_lines=8):
        L = ["WHAT HAS BEEN TRIED ON THIS PROMPT SO FAR (mean outcome = mean terminal reward of trajectories using that wording):"]
        rows = sorted(range(len(self.wordings)), key=lambda i: (-(self.s[i] / self.n[i]) if self.n[i] else 1e9, i))
        shown = 0
        for i in rows:
            if shown >= max_lines: break
            tag = " <- the wording at this node" if i == cur_wi else ""
            if self.n[i]: L.append(f'  "{self.wordings[i][:90]}": tried {self.n[i]}x, mean outcome {self.s[i]/self.n[i]:.3f}, best {self.best[i]:.3f}{tag}')
            else: L.append(f'  "{self.wordings[i][:90]}": not tried yet{tag}')
            shown += 1
        if path_actions:
            L.append("PATH TO THIS NODE: " + " -> ".join(f'w{int(v)}@CFG{float(c):g}' for v, c, *_ in path_actions))
        done = [e for e in self.edits if e["fixed"] or e["broke"]]
        if done:
            L.append("EDITS ALREADY MADE ON THIS PROMPT AND WHAT THEY CHANGED:")
            for e in done[-3:]:
                L.append(f'  {e["operator"]}: fixed elements {e["fixed"] or "none"}, broke {e["broke"] or "none"} '
                         f'({e["n_present_before"]} -> {e["n_present_after"]} present)')
        return "\n".join(L)

class WordingBase:
    """Append-only cross-prompt record of which operator fixed which kind of checklist element."""
    def __init__(self, path, frozen=False):
        self.path = path; self.frozen = frozen; self._lock = threading.Lock(); self.rows = []
        if os.path.exists(path):
            for l in open(path):
                try: self.rows.append(json.loads(l))
                except Exception: continue

    def add(self, prompt_id, operator, element_types_fixed, element_types_broke, wording_before, wording_after, score_delta=None):
        if self.frozen: return
        row = {"prompt_id": prompt_id, "operator": operator, "fixed_types": element_types_fixed, "broke_types": element_types_broke,
               "wording_before": wording_before[:200], "wording_after": wording_after[:200], "score_delta": score_delta}
        with self._lock:
            self.rows.append(row)
            with open(self.path, "a") as f: f.write(json.dumps(row) + "\n")

    def stats(self):
        """(element_type, operator) -> (n_fixed, n_attempts_touching_that_type)."""
        fixed = defaultdict(int); tried = defaultdict(int)
        for r in self.rows:
            op = r.get("operator", "?")
            for t in r.get("fixed_types", []): fixed[(t, op)] += 1; tried[(t, op)] += 1
            for t in r.get("broke_types", []): tried[(t, op)] += 1
        return fixed, tried

    def render(self, element_types=None, max_lines=5, min_n=2):
        fixed, tried = self.stats()
        if not tried: return ""
        rows = []
        for key, n in tried.items():
            t, op = key
            if element_types and t not in element_types: continue
            if n < min_n: continue
            rows.append((fixed[key] / n, fixed[key], n, t, op))
        if not rows: return ""
        rows.sort(reverse=True)
        L = ["WHAT HAS WORKED ON OTHER PROMPTS (operator success at turning that element type into `present`):"]
        for rate, f, n, t, op in rows[:max_lines]: L.append(f"  {t}: the `{op}` operator fixed it {f}/{n} times")
        return "\n".join(L)

if __name__ == "__main__":
    import sys
    b = WordingBase(sys.argv[1] if len(sys.argv) > 1 else "/data/ygu/runs/vlm_decider_sd35base_2026-10-16/state/wording_base.jsonl")
    print(f"{len(b.rows)} rows"); print(b.render(max_lines=12, min_n=1) or "(no aggregate yet)")
