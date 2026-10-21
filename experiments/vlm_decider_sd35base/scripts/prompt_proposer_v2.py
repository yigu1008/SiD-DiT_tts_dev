#!/usr/bin/env python3
"""Proposer v2: maximise wording DIVERSITY subject to not corrupting the constraint.

v1 was built on the wrong objective and the judged data says so. It forbade style/lighting language and demanded
full content-word retention, which yielded only **56% valid -> 2.67 distinct wordings vs the shipped bank's 4.00**,
and the result was hpsv3 **-0.193 [-0.330, -0.078]** with **no** compensating gain on the faithfulness judges
(vqascore -0.009, qwen +0.006, both inconclusive). The lesson measured there was that in reward-guided search,
proposal DIVERSITY beats proposal CORRECTNESS: selection already discards a bad variant (75% of generations used a
non-base wording and the corrupt ones simply lost), but it cannot invent a proposal that is missing.

So v2 keeps only the checks that protect the thing under test, and drops the rest:
  KEEP    numeral multiset preserved; negation cues not reduced; base content words retained; no generator
          artifacts; fluency (no repeated 5-gram, no telegraphic fragment list).
  DROP    the style/lighting/camera prohibition -- HPSv3 is an aesthetic preference model and those cues are
          legitimately part of what it rewards.
  ADD     many short, varied instructions x temperatures, accepting EVERY distinct passing candidate rather than
          one per slot, to push the bank well past 4 wordings and widen the action space itself.
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time, urllib.request
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from prompt_proposer import post, sanitize, NEG, ARTIFACTS, PMAP

# v2's own numeral / content-word rules. v1's were the BOTTLENECK, not the generator: on the 9-prompt cohort they
# rejected 26-28 of 32 candidates for "dropped" words and produced 0 valid wordings for q0041, because
#  (a) the stopword list was tiny, so verbatim retention of FUNCTION words ("which", "each", "other", "there") was
#      demanded of every paraphrase, and
#  (b) numerals were compared as an exact multiset, so "none of which are adjacent to ONE another" or "THE ONE with
#      a long hemline" added a spurious 1 and failed a prompt whose only real count was "four".
# Fix: a real function-word list, and count only numbers that actually count -- digits and two..twelve. The word
# "one" is overwhelmingly pronominal in these paraphrases; the content-word check still protects the noun it
# would have modified, and the failure mode actually measured in the shipped cache was two -> three, which the
# >= 2 rule still catches.
STOP2 = {
    "a","an","the","of","with","and","in","on","is","are","there","where","to","it","that","this","at","by","for",
    "as","be","but","or","its","their","which","each","other","another","one","from","into","when","than","them",
    "they","then","while","also","very","some","such","these","those","has","have","having","been","being","was",
    "were","its","his","her","our","your","who","whom","whose","what","how","all","any","both","more","most","much",
    "many","over","under","out","off","up","down","about","above","below","between","through","during","after",
    "before","again","once","here","both","same","so","too","can","will","just","now","only","own","than","upon",
}
NUMW2 = {"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}


def numerals(t: str) -> list[int]:
    t = str(t).lower()
    out = [int(m.group(1)) for m in re.finditer(r"\b(\d+)\b", t)]
    for w, v in NUMW2.items(): out += [v] * len(re.findall(r"\b" + w + r"\b", t))
    return sorted(out)


def content_words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(t).lower()) if len(w) > 3 and w not in STOP2}


INSTRUCTIONS = [
    "Restate the prompt so the specific thing it asks for is impossible to miss. Name every object plainly.",
    "Describe the very same scene in concrete visual terms, as if telling someone exactly what to draw.",
    "Rewrite the prompt so the element an image generator is most likely to get wrong comes FIRST in the sentence.",
    "Rewrite the prompt as a vivid photograph description. You may add lighting, time of day or camera language.",
    "Rewrite the prompt in a richer, more descriptive register. You may add texture, material and colour detail.",
    "Rewrite the prompt as a single long, flowing sentence with more visual specificity.",
    "Rewrite the prompt tersely, keeping only what must appear in the image.",
    "Rewrite the prompt emphasising the spatial arrangement of the things in it.",
]
SYSTEM = ("You rewrite text-to-image prompts. You may add visual, lighting, material or stylistic detail, but you "
          "NEVER change what is in the scene: never change a number, never remove or soften a negation, never drop "
          "or rename an object the prompt names, never add a new object. Output exactly one rewritten prompt as a "
          "single line of plain text, with no preamble, no quotes and no commentary.")


def validate(base, cand, accepted):
    c = cand
    if not c: return False, "empty"
    low = c.lower()
    for a in ARTIFACTS:
        if a in low: return False, f"artifact {a!r}"
    if any(ord(ch) < 32 for ch in c): return False, "control char"
    if numerals(c) != numerals(base): return False, f"numerals {numerals(base)} -> {numerals(c)}"
    toks = set(re.findall(r"[a-z]+", low))
    missing = {w for w in content_words(base) - toks
               if not any(x.startswith(w[: max(4, len(w) - 2)]) for x in toks)}
    if missing: return False, f"dropped {sorted(missing)}"
    nb, nc = len(NEG.findall(base)), len(NEG.findall(c))
    if nb and nc < nb: return False, f"negation {nb} -> {nc}"
    if len(c) > max(320, int(len(base) * 3.5)): return False, "too long"
    if len(c) < max(8, int(len(base) * 0.6)): return False, "too short"
    if c.strip().lower() == base.strip().lower(): return False, "identical to base"
    t = re.findall(r"[a-z]+", low)
    for i in range(len(t) - 4):
        if low.count(" ".join(t[i:i + 5])) > 1: return False, "repeated clause"
    if c.count(";") > 1: return False, "telegraphic"
    frags = [f.strip() for f in re.split(r"[;,]", c) if f.strip()]
    if frags and sum(1 for f in frags if len(f.split()) <= 2) > max(1, len(frags) // 3): return False, "fragment list"
    for x in accepted:
        if c.strip().lower() == x.strip().lower(): return False, "duplicate"
    return True, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True); p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=10, help="distinct non-base wordings wanted per prompt")
    p.add_argument("--vlm_server", default="http://127.0.0.1:5151"); p.add_argument("--report", default=None)
    a = p.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}
    assert json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read())["status"] == "ok"
    temps = [0.0, 0.5, 0.8, 1.0]
    cache = {}; rep = []
    for pid in ids:
        base = pmap[pid]; acc = []; tried = 0; rej = {}
        for t in temps:
            for ins in INSTRUCTIONS:
                if len(acc) >= a.target: break
                tried += 1
                r = post(a.vlm_server + "/generate", {"system": SYSTEM, "text": f"{ins}\n\nThe prompt to rewrite:\n{base}",
                                                       "images_b64": [], "max_new_tokens": 110, "temperature": t})
                cand = sanitize(r["text"]); ok, why = validate(base, cand, acc)
                if ok: acc.append(cand)
                else: rej[why.split()[0]] = rej.get(why.split()[0], 0) + 1
            if len(acc) >= a.target: break
        cache[base] = [base] + acc
        rep.append({"prompt_id": pid, "base": base, "n_valid": len(acc), "tried": tried,
                    "reject_reasons": rej, "variants": [base] + acc})
        print(f"[{pid}] {len(acc)} distinct valid wordings from {tried} calls; rejects {rej}", flush=True)
        for v in acc: print(f"      {v[:135]}", flush=True)
    Path(a.out).write_text(json.dumps(cache, indent=1))
    if a.report: Path(a.report).write_text(json.dumps(rep, indent=1))
    n = [len(r["variants"]) for r in rep]
    print(f"\nwrote {a.out}: {len(cache)} prompts, mean {sum(n)/len(n):.2f} DISTINCT wordings per prompt "
          f"(shipped bank 4.00, v1 proposer 2.67); action space = that x 7 CFG levels")


if __name__ == "__main__":
    main()
