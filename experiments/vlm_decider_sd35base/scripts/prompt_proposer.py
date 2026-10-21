#!/usr/bin/env python3
"""A constraint-PRESERVING prompt proposer, replacing the cached rewrite bank.

Audit of the shipped cache (`shared_rewrites_cache.json`, 200 prompts x 3 variants) found, across 600 variants:
  * **46.0%** corrupt the prompt -- the numeral multiset changes ("one bird, two flowers" -> "three flowers")
    or a base content word is dropped ("every chair empty" -> "No students in an empty classroom");
  * **20.2%** contain generator artifacts: 86 carry `/no_think` and 35 carry `Original prompt:`. Root cause is
    `precompute_sd35_rewrites.py:156`, which appends " /no_think" to the USER message; Qwen echoed it and the
    output was cached unsanitized. `sanitize_rewrite_text` exists but is NOT called on the path the arms use
    (`generate_variants` -> `_read_legacy_cache_entry` -> `_dedup_variants`, which only does `str(v)`), so this
    text reaches the T5/CLIP encoders verbatim;
  * **56.2%** add only style/lighting decoration ("Late afternoon sunlight...", "Moody and slightly cluttered")
    rather than reinforcing the constraint the prompt actually tests;
  * of the 117 variants on the 39 negation prompts, **35 lose the negation** ("a red harp without any strings" ->
    "Minimalist harp with bare strings").
And the search consumes them: **58.5%** of selected actions across all arms used a non-base wording.

This proposer instead asks for restatements that make the tested constraint MORE explicit, and hard-validates every
candidate. A slot that cannot produce a valid candidate falls back to the base prompt, so the bank shrinks rather
than being poisoned -- a smaller honest bank beats a larger corrupt one.
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time, urllib.request
from pathlib import Path

PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
ARTIFACTS = ("/no_think", "<think>", "</think>", "original prompt", "rewritten prompt", "here is the rewrite",
             "as an ai", "sure,", "```", "prompt:")
STOP = {"a", "an", "the", "of", "with", "and", "in", "on", "is", "are", "there", "where", "to", "it",
        "that", "this", "at", "by", "for", "as", "be", "but", "or", "its", "their"}
NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "pair": 2, "couple": 2, "single": 1}
NEG = re.compile(r"\b(no|not|without|never|empty|devoid|absent|neither|nor|none|lacks?|lacking|"
                 r"doesn't|don't|isn't|aren't|hasn't|haven't|bare|barren|unoccupied|vacant)\b", re.I)

SLOTS = [
    ("explicit", "Restate the prompt so the specific thing it asks for is impossible to miss. Keep every object, "
                 "every number and every absence exactly as given, and name them plainly. Do not add lighting, "
                 "mood, camera or style words."),
    ("concrete", "Describe the very same scene in concrete visual terms, as if telling someone exactly what to draw. "
                 "Preserve every count and every stated absence precisely. Do not introduce objects that are not "
                 "mentioned, and do not add lighting, mood, camera or style words."),
    ("frontload", "Identify the single element of this prompt that an image generator is most likely to get wrong -- a "
                  "count, a comparison, a negation or a spatial relation -- and rewrite the prompt so that element "
                  "comes FIRST, at the very start of the sentence. Say it once, not twice. Keep every object, number "
                  "and absence exactly as given. Do not add lighting, mood, camera or style words."),
]
SYSTEM = ("You rewrite text-to-image prompts. You never change what is in the scene: never change a number, never "
          "remove or soften a negation, never drop or rename an object, never add an object. You output exactly one "
          "rewritten prompt as a single line of plain text, with no preamble, no quotes and no commentary.")


def post(url, body, timeout=300, tries=30, sleep=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:60]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")


def numerals(t: str) -> list[int]:
    t = t.lower(); out = [int(m.group(1)) for m in re.finditer(r"\b(\d+)\b", t)]
    for w, v in NUMWORDS.items(): out += [v] * len(re.findall(r"\b" + w + r"\b", t))
    return sorted(out)


def content_words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 3 and w not in STOP}


def sanitize(text: str) -> str:
    """What sanitize_rewrite_text should have done on the cache path, plus the artifacts it misses."""
    t = re.sub(r"<think>.*?</think>", " ", str(text), flags=re.DOTALL)
    t = re.sub(r"(?i)\boriginal prompt\s*:?", " ", t)
    t = re.sub(r"(?i)\brewritten prompt\s*:?", " ", t)
    t = t.replace("/no_think", " ").replace("�", "")
    t = t.split("\n")[0]
    t = re.sub(r"\s+", " ", t).strip().strip("`\"' ")
    return t


def validate(base: str, cand: str, accepted: list[str]) -> tuple[bool, str]:
    c = cand
    if not c: return False, "empty"
    low = c.lower()
    for a in ARTIFACTS:
        if a in low: return False, f"artifact {a!r}"
    if any(ord(ch) < 32 for ch in c): return False, "control char"
    if numerals(c) != numerals(base): return False, f"numerals {numerals(base)} -> {numerals(c)}"
    missing = content_words(base) - set(re.findall(r"[a-z]+", low))
    # allow a base word to be matched by a longer form of itself (bloom -> blooming)
    missing = {w for w in missing if not any(x.startswith(w[:max(4, len(w) - 2)]) for x in re.findall(r"[a-z]+", low))}
    if missing: return False, f"dropped {sorted(missing)}"
    nb, nc = len(NEG.findall(base)), len(NEG.findall(c))
    if nb and nc < nb: return False, f"negation cues {nb} -> {nc}"
    if len(c) > max(240, int(len(base) * 2.5)): return False, f"too long ({len(c)})"
    if len(c) < max(8, int(len(base) * 0.7)): return False, f"too short ({len(c)})"
    if c.strip().lower() == base.strip().lower(): return False, "identical to base"
    for x in accepted:
        if c.strip().lower() == x.strip().lower(): return False, "duplicate of another slot"
    toks = re.findall(r"[a-z]+", low)
    for i in range(len(toks) - 4):                       # repeated 5-gram = the model padded by duplicating a clause
        g = " ".join(toks[i:i + 5])
        if low.count(g) > 1: return False, f"repeated clause {g!r}"
    if c.count(";") > 1: return False, "telegraphic (multiple semicolons)"
    frags = [f.strip() for f in re.split(r"[;,]", c) if f.strip()]
    if frags and sum(1 for f in frags if len(f.split()) <= 2) > max(1, len(frags) // 3):
        return False, "telegraphic (fragment list)"
    return True, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", required=True); p.add_argument("--out", required=True)
    p.add_argument("--vlm_server", default="http://127.0.0.1:5151")
    p.add_argument("--tries", type=int, default=6); p.add_argument("--report", default=None)
    a = p.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}
    assert json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read())["status"] == "ok"
    temps = [0.0, 0.3, 0.6, 0.9, 1.0, 1.0]
    cache = {}; rep = []
    for pid in ids:
        base = pmap[pid]; accepted = []; slotlog = []
        for name, instr in SLOTS:
            got = None; why = []
            for t in range(a.tries):
                r = post(a.vlm_server + "/generate", {"system": SYSTEM,
                         "text": f'{instr}\n\nThe prompt to rewrite:\n{base}',
                         "images_b64": [], "max_new_tokens": 90,
                         "temperature": temps[min(t, len(temps) - 1)]})
                cand = sanitize(r["text"])
                ok, msg = validate(base, cand, accepted)
                if ok: got = cand; why.append(f"attempt {t+1}: ok"); break
                why.append(f"attempt {t+1} (T={temps[min(t,len(temps)-1)]}): {msg}")
            if got: accepted.append(got)
            slotlog.append({"slot": name, "accepted": got, "trace": why})
        variants = [base] + accepted
        while len(variants) < 4: variants.append(base)          # honest shrink, never a corrupt filler
        cache[base] = variants
        rep.append({"prompt_id": pid, "base": base, "n_valid": len(accepted), "slots": slotlog, "variants": variants})
        print(f"[{pid}] {len(accepted)}/3 valid", flush=True)
        for s in slotlog:
            print(f"    {s['slot']:<9} {'OK ' if s['accepted'] else 'FAIL'} {s['accepted'] or s['trace'][-1]}", flush=True)
    Path(a.out).write_text(json.dumps(cache, indent=1))
    if a.report: Path(a.report).write_text(json.dumps(rep, indent=1))
    nv = sum(r["n_valid"] for r in rep)
    print(f"\nwrote {a.out}: {len(cache)} prompts, {nv}/{3*len(ids)} valid non-base variants "
          f"({100*nv/(3*len(ids)):.0f}%); slots that could not be filled fall back to the base prompt")


if __name__ == "__main__":
    main()
