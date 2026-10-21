#!/usr/bin/env python3
"""Deliberate discrete prompt optimizer (offline, per prompt) on base SD3.5-Large -> arm `opt_prompt`.

Loop (rounds R, default 4):
  0. checklist  : VLM (text only) decomposes the ORIGINAL prompt into typed elements (object/count/attribute/relation/style/scene).
  1. sample     : encode the current wording, plain 28-step Euler sample at baseline CFG (seed 42)  -> image_r
  2. readout    : VLM grades every checklist element on image_r as present / missing / wrong (+ note)        [perceived state]
  3. propose    : VLM (text only) writes K candidate wordings, each = ONE typed operator on the current wording aimed at a
                  missing/wrong element: emphasize | reorder | clarify | drop_clause | revert
  4. guard      : Jev-Omni yes/no (text): "candidate describes exactly the same scene as the ORIGINAL (nothing added, nothing removed)?"
                  P(yes) < --guard_thresh -> discarded.  No reward is consulted anywhere in the loop.
  5. decide     : Jev-Omni choice over surviving candidates + "keep current wording", state = original + checklist statuses + current
                  wording, media = image_r.  "keep" (or all present) ends the loop.
Output image = the round whose checklist has the most `present` elements (ties -> latest round), i.e. a completeness criterion, not IR.
ImageReward of the output is stored as `search_score` for the exporter only.  All wordings/embeddings are cached per prompt; every
round (image, statuses, candidates, guard P(yes), Jev probabilities, choice) is logged in records.jsonl and rounds/<pid>_r<k>.png.
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, os, re, sys, time, urllib.request
from pathlib import Path

SID = "/home/ygu/project/SiD-DiT_tts_dev"; PMAP = "/data/ygu/reward_orthogonality/mcts/prompts/prompt_map.csv"
CFG_BANK = ["3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "7.0"]; BASELINE_CFG = "4.5"
OPERATORS = {"emphasize": "restate or stress an element already in the original (e.g. make the count or the spatial relation explicit, repeat the key noun)",
             "reorder": "move the missing element earlier in the sentence so the model attends to it first",
             "clarify": "rephrase an ambiguous clause of the original into plainer visual language WITHOUT adding new content",
             "drop_clause": "remove or shorten a clause that is pulling the image away from the missing element (the element itself must stay)",
             "revert": "return to the ORIGINAL prompt wording"}

def post(url, body, timeout=600, tries=60, sleep=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:
            print(f"[post] {url} failed ({type(e).__name__}: {str(e)[:80]}); retry {i+1}/{tries}", flush=True); time.sleep(sleep)
    raise RuntimeError(f"{url} unreachable")

def _balanced(text, open_ch, close_ch):
    """Yield every balanced open_ch..close_ch span in text (string-aware)."""
    stack = []; in_str = False; esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True; continue
        if ch == open_ch:
            stack.append(i)
        elif ch == close_ch and stack:
            j = stack.pop(); yield text[j:i + 1]        # every nesting level, innermost first

def jtxt(raw):
    """First JSON value in a model reply, with repair: strict parse, then trailing-comma repair, then salvage of the
    individually-parsable {...} objects (long arrays get truncated or mis-delimited by the VLM)."""
    m = re.search(r"(\[.*\]|\{.*\})", raw, re.S)
    span = m.group(1) if m else raw
    try: return json.loads(span)
    except Exception: pass
    try: return json.loads(re.sub(r",\s*([}\]])", r"\1", span))
    except Exception: pass
    first = min([i for i in (raw.find("["), raw.find("{")) if i >= 0], default=0)
    tail = raw[first:]                                   # truncated replies have no closing bracket: scan the whole tail
    objs = []
    for o in _balanced(tail, "{", "}"):
        try: objs.append(json.loads(re.sub(r",\s*([}\]])", r"\1", o)))
        except Exception: continue
    if objs:
        keyed = [o for o in objs if "grades" in o]
        if keyed: return keyed[0]
        inner = [o for o in objs if ("id" in o and "status" in o) or "wording" in o or "text" in o]
        if inner:
            mo = re.search(r'"observations"\s*:\s*"((?:[^"\\]|\\.)*)"', tail)
            if mo and any("status" in o for o in inner): return {"observations": mo.group(1), "grades": inner}
            return inner
        return objs if len(objs) > 1 else objs[0]
    raise ValueError("no parsable JSON in reply: " + raw[:300])

def _retry_json(call, what="reply"):
    """Run call(temperature) until its JSON parses; greedy first, then two sampled retries."""
    last = None
    for t in (0.0, 0.7, 1.0):
        try: return call(t)
        except Exception as e:
            last = e; print(f"[json] {what} unparsable at temperature {t} ({str(e)[:90]}); resampling", flush=True)
    raise last

def b64(img, px=896):
    im = img.resize((px, px)); buf = io.BytesIO(); im.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()

SYS_CHECK = ("You are a meticulous visual-content auditor for text-to-image generation. You decompose prompts into atomic, checkable visual elements and "
             "grade images strictly against them. Answer only with the requested JSON.")

def extract_checklist(vlm, prompt):
    text = (f'PROMPT: "{prompt}"\n\nDecompose this prompt into atomic visual statements that an image either satisfies or not. Each statement must be '
            f'SELF-CONTAINED and name its subject (write "exactly two birds", "the kitten is on the left side", "the higher-flying bird has a long tail", '
            f'never bare words like "two" or "left"). Include every count as "exactly N <object>", every left/right or higher/lower relation, every attribute, '
            f'and the IMPLIED contrasts of comparisons (e.g. "the other dresses have shorter hemlines than the most vibrant one"). Types: object, count, '
            f'attribute, spatial_relation, action_relation, comparison, negation, style, scene, text. 3-10 statements. Respond with JSON only: '
            f'[{{"id": 1, "type": "...", "text": "..."}}, ...]')
    out = {}
    def _call(temp):
        r = post(vlm + "/generate", {"system": SYS_CHECK, "text": text, "images_b64": [], "max_new_tokens": 500, "temperature": temp}); out["s"] = r["seconds"]; return jtxt(r["text"])
    el = _retry_json(_call, what="checklist")
    el = [e for e in (el if isinstance(el, list) else [el]) if isinstance(e, dict) and "text" in e]
    return [{"id": int(e.get("id", i + 1)), "type": str(e.get("type", "")), "text": str(e["text"])} for i, e in enumerate(el)], out.get("s", 0.0)

def readout(vlm, prompt, elements, img, jev=None, seed=0):
    el = "\n".join(f'  {e["id"]}. [{e["type"]}] {e["text"]}' for e in elements)
    text = (f'ORIGINAL PROMPT: "{prompt}"\n\nCHECKLIST:\n{el}\n\nStep 1 (observations): look carefully and write what the image actually shows that is relevant '
            f'to the checklist: count each kind of object explicitly, state each object\'s position (left/right/higher/lower) and the attribute in question for EACH object '
            f'separately (e.g. "left bird: tail short; right bird: tail short"). Step 2 (grades): for every checklist id give status "present" ONLY if your '
            f'observations verify it exactly, "wrong" if the image shows it incorrectly (wrong count, wrong side, attribute on the wrong object, no visible '
            f'difference where a difference is required), or "missing" if not shown. When unsure, do NOT say present. '
            f'Respond with JSON only: {{"observations": "<=80 words", "grades": [{{"id": 1, "status": "present|wrong|missing", "note": "<=12 words"}}, ...]}}')
    out = {}
    def _call(temp):
        r = post(vlm + "/generate", {"system": SYS_CHECK, "text": text, "images_b64": [b64(img)], "max_new_tokens": 600, "temperature": temp}); out["s"] = r["seconds"]; return jtxt(r["text"])
    try: j = _retry_json(_call, what="readout")
    except Exception as exc:
        print(f"[readout] unparsable after retries ({str(exc)[:80]}); all elements graded unknown", flush=True)
        return [{"id": e_["id"], "status": "unknown", "note": "readout unparsable", "vlm_status": "unknown", "jev_p_yes": None} for e_ in elements], out.get("s", 0.0), ""
    secs = out.get("s", 0.0)
    obs = str(j.get("observations", "")) if isinstance(j, dict) else ""
    gr = j.get("grades", j) if isinstance(j, dict) else j
    by = {int(g["id"]): {"status": str(g.get("status", "")).lower().strip(), "note": str(g.get("note", ""))} for g in gr}
    out = []
    for e in elements:
        st = by.get(e["id"], {}).get("status", "unknown"); note = by.get(e["id"], {}).get("note", ""); jp = None
        if jev is not None:
            state = (f'ORIGINAL PROMPT: "{prompt}"\nVLM observations of the attached image: {obs}\nSTATEMENT TO VERIFY: "{e["text"]}"')
            rj = post(jev + "/ask", {"state": state, "question": "Does the attached image satisfy the STATEMENT exactly (correct count, side, and attribute on the right object)?",
                                     "options": ["Yes", "No"], "image_b64": b64(img), "orders": 2, "seed": seed + e["id"], "tag": "optprompt/verify"})
            jp = float(rj["probs"][0]); secs += rj["ms"] / 1000
            if st == "present" and jp < 0.5: st = "wrong"; note = (note + " | Jev disagrees").strip(" |")
        out.append({"id": e["id"], "status": st, "note": note, "vlm_status": by.get(e["id"], {}).get("status", "unknown"), "jev_p_yes": jp})
    return out, secs, obs

ALLOWED_WORDS = {"exactly", "only", "single", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "the", "other", "others", "remaining",
                 "rest", "each", "both", "all", "none", "no", "not", "never", "than", "more", "less", "most", "least", "shorter", "longer", "smaller", "larger",
                 "bigger", "higher", "lower", "clearly", "visibly", "distinct", "distinctly", "different", "differently", "separate", "separately", "apart", "while",
                 "whereas", "and", "but", "with", "which", "that", "whose", "is", "are", "has", "have", "having", "being", "a", "an", "of", "in", "on", "at", "to",
                 "its", "their", "it", "they", "them", "there", "this", "these", "those", "another", "same", "such", "very", "so", "as", "from", "by", "for", "into",
                 "onto", "over", "under", "above", "below", "between", "among", "beside", "next", "side", "left", "right", "top", "bottom", "front", "behind", "far",
                 "away", "touching", "touch", "contact", "gap", "space", "spaced", "evenly", "visible", "shown", "seen", "wearing", "worn", "around", "wrist", "wrists",
                 "sitting", "sits", "sit", "seated", "be", "been", "does", "do", "did", "than", "any", "every", "else", "also", "too", "well", "still", "her", "his", "him", "she", "he", "hers", "itself", "themselves", "own"}
STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "with", "and", "or", "but", "is", "are", "was", "were", "be", "being", "been", "has", "have", "having", "its", "their",
        "it", "they", "them", "there", "this", "that", "these", "those", "which", "who", "whose", "while", "whereas", "as", "by", "for", "from", "into", "onto", "each", "other",
        "others", "one", "ones", "none", "not", "no", "all", "both", "any", "every", "very", "so", "than", "more", "most", "less", "least", "such", "same", "also", "too"}

def _toks(t):
    import re as _re
    return [w for w in _re.findall(r"[a-z0-9']+", t.lower())]

def _stem(w):
    if w.endswith("ies") and len(w) > 5: return w[:-3] + "y"
    if w.endswith("sses"): return w[:-2]                      # dresses -> dress
    if w.endswith("ss"): return w                             # dress, glass: keep
    for suf in ("es", "s", "ing", "ed", "ly", "er", "est"):
        if w.endswith(suf) and len(w) - len(suf) >= 4: return w[: -len(suf)]
    return w

def lexical_check(prompt, cand):
    """(ok, reason). Retention: every content word of the original must appear (stem match). No-new-content: every content word of the
    candidate must come from the original or the allowed emphasis vocabulary."""
    ot = {_stem(w) for w in _toks(prompt)}; oc = {_stem(w) for w in _toks(prompt) if w not in STOP}
    ct = [w for w in _toks(cand)]; cs = {_stem(w) for w in ct}
    missing = [w for w in oc if w not in cs]
    new = sorted({w for w in ct if w not in STOP and _stem(w) not in ot and w not in ALLOWED_WORDS and _stem(w) not in {_stem(a) for a in ALLOWED_WORDS}})
    if missing: return False, f"drops: {missing}"
    if new: return True, f"adds: {new} (advisory; Jev guard decides additions)"   # v5: lexical filter enforces RETENTION only
    return True, ""

def propose(vlm, prompt, wording, elements, statuses, k, guidance=""):
    bad = [e for e, s in zip(elements, statuses) if s["status"] in ("missing", "wrong")]
    el = "\n".join(f'  {e["id"]}. [{e["type"]}] {e["text"]}  -> {s["status"]}' for e, s in zip(elements, statuses))   # statuses only: notes/observations are NOT shown
    ops = "\n".join(f"  - {k_}: {v}" for k_, v in OPERATORS.items())
    text = (f'ORIGINAL PROMPT (the only target; it defines ALL allowed content): "{prompt}"\nCURRENT WORDING sent to the image model: "{wording}"\n\n'
            f'CHECKLIST STATUS on the latest image:\n{el}\n\nWrite {k} alternative wordings. Each applies exactly ONE operator to the CURRENT wording, aimed at one '
            f'missing/wrong element (ids {[e["id"] for e in bad]}):\n{ops}\n\nVOCABULARY RULE: use only words that already appear in the ORIGINAL PROMPT plus function '
            f'words and emphasis words such as "exactly", "only", "the other", "the remaining", "shorter/longer", "none of", "clearly", "visibly", "distinct". '
            f'Never name colors, materials, positions or any attribute that the ORIGINAL does not state, even if you believe the image shows them. Never drop an '
            f'element of the ORIGINAL. Make implied contrasts explicit when a comparison/differentiation element is wrong (example: for "four dresses; the one with a long '
            f'hemline is the most vibrant" an emphasize edit is "Exactly one of the four dresses on the hanger has a long hemline and the other three are clearly shorter; '
            f'that long dress is the most vibrant."). Keep each wording one or two sentences; vary the operators across candidates. '
            + (f"\n\n{guidance}\n" if guidance else "")
            + f'Respond with JSON only: [{{"wording": "...", "operator": "emphasize|reorder|clarify|drop_clause|revert", "target_ids": [..], "rationale": "<=15 words"}}, ...]')
    out = []; seen = set(); secs = 0.0; raw_n = 0
    for attempt, temp in enumerate((0.0, 0.7, 0.9)):          # greedy first; if fewer than 2 usable candidates, resample with temperature
        r = post(vlm + "/generate", {"system": SYS_CHECK, "text": text, "images_b64": [], "max_new_tokens": 700, "temperature": temp}); secs += r["seconds"]
        try: c = jtxt(r["text"]); c = c if isinstance(c, list) else [c]
        except Exception as exc: print(f"[propose] unparsable at temperature {temp} ({str(exc)[:70]})", flush=True); c = []
        raw_n += len(c)
        for x in c:
            w = " ".join(str(x.get("wording", "")).split()) if isinstance(x, dict) else ""
            if not w or w in seen or w == wording: continue
            seen.add(w); ok, why = lexical_check(prompt, w)
            out.append({"wording": w, "operator": str(x.get("operator", "")), "target_ids": x.get("target_ids", []), "rationale": str(x.get("rationale", "")), "lexical_ok": ok, "lexical_reason": why, "attempt": attempt})
        if sum(1 for o in out if o["lexical_ok"]) >= 2: break
    if not out: print(f"[propose] no usable candidates after 3 attempts ({raw_n} raw)", flush=True)
    return out, secs

GUARD_STATE = ('ORIGINAL PROMPT: "{prompt}"\nCANDIDATE WORDING: "{cand}"\n\nA candidate is FAITHFUL if it describes the same scene as the original: it may '
               'rephrase, reorder, stress, or make EXPLICIT what the original logically implies (e.g. "the one with a long hemline" implies exactly one such dress '
               'and that the others are shorter; "two birds, the higher one has a long tail" implies the lower one does not). It is UNFAITHFUL if it introduces '
               'any object, attribute, color, style, mood, lighting, setting or count the original neither states nor implies, if it drops an element of the '
               'original, or if it contradicts the original.')

def guard(jev, prompt, cand, seed):
    r = post(jev + "/ask", {"state": GUARD_STATE.format(prompt=prompt, cand=cand), "question": "Is the CANDIDATE WORDING faithful to the ORIGINAL PROMPT under this definition?",
                            "options": ["Yes", "No"], "orders": 2, "seed": seed, "tag": "optprompt/guard"})
    return float(r["probs"][0]), float(r["spread"][0]), r["ms"]

def decide(jev, prompt, wording, elements, statuses, cands, img, seed):
    el = "\n".join(f'  {e["id"]}. {e["text"]} -> {s["status"]}' + (f' ({s["note"]})' if s["note"] else "") for e, s in zip(elements, statuses))
    state = (f'ORIGINAL PROMPT (the only target): "{prompt}"\nCURRENT WORDING: "{wording}"\nCHECKLIST on the latest image (attached):\n{el}\n'
             f"The image model will be re-run with the chosen wording at the same seed; the goal is an image in which every checklist element is present and correct.")
    opts = [c["wording"] for c in cands] + ["Keep the current wording unchanged"]
    r = post(jev + "/ask", {"state": state, "question": "Which wording should be sent to the image model next?", "options": opts, "image_b64": b64(img),
                            "orders": 3, "seed": seed, "tag": "optprompt/decide"})
    return r["argmax"], r["probs"], r["spread"], r["ms"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default=None, help="default: basename of --run_dir (so versions never collide in the judge)"); p.add_argument("--ids", required=True); p.add_argument("--run_dir", required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--seed", type=int, default=42); p.add_argument("--rounds", type=int, default=4); p.add_argument("--k", type=int, default=4)
    p.add_argument("--guard_thresh", type=float, default=0.7); p.add_argument("--reward_server", required=True)
    p.add_argument("--vlm_server", default="http://127.0.0.1:5151"); p.add_argument("--jev_server", default="http://127.0.0.1:5161")
    a = p.parse_args(); a.arm = a.arm or Path(a.run_dir).name
    os.environ.update({"HF_HUB_CACHE": "/data/ygu/.cache/hub", "HF_HOME": "/data/ygu/.cache/hub", "HF_HUB_OFFLINE": "1",
                       "OFFLOAD_TEXT_ENCODER_AFTER_ENCODE": "1", "REWARD_SERVER_URL": a.reward_server, "HPSV3_SCORE_STYLE": "raw"})
    sys.path.insert(0, SID); os.chdir(SID)
    import torch
    import sampling_unified_sd35 as su
    for url in (a.vlm_server, a.jev_server):
        h = json.loads(urllib.request.urlopen(url + "/health", timeout=10).read()); assert h.get("status") == "ok", (url, h)
    sargs = su.normalize_paths(su.parse_args(["--backend", "sd35_base", "--search_method", "greedy", "--reward_backend", "imagereward", "--no_qwen", "--n_variants", "0",
        "--steps", "28", "--seed", str(a.seed), "--width", "1024", "--height", "1024", "--out_dir", a.run_dir, "--cfg_scales", *CFG_BANK, "--baseline_cfg", BASELINE_CFG]))
    assert getattr(sargs, "euler_sampler", False) and sargs.model_id == "stabilityai/stable-diffusion-3.5-large"
    run_dir = Path(a.run_dir); (run_dir / "rounds").mkdir(parents=True, exist_ok=True); (run_dir / "checklists").mkdir(exist_ok=True)
    ids = [l.strip() for l in open(a.ids) if l.strip()]; ids = ids[: a.limit] if a.limit else ids
    pmap = {r["prompt_id"]: r["prompt"] for r in csv.DictReader(open(PMAP))}
    recp = run_dir / "records.jsonl"; done = {json.loads(l)["prompt_id"] for l in open(recp)} if recp.exists() else set()
    todo = [i for i in ids if i not in done]; print(f"[{a.arm}] {len(todo)} prompts to do ({len(done)} done)", flush=True)
    if not todo: return
    ctx = su.load_pipeline(sargs); rm = su.load_reward_model(sargs, ctx.device)
    (run_dir / "config.json").write_text(json.dumps({"arm": a.arm, "backend": "sd35_base", "model_id": sargs.model_id, "dtype": str(sargs.dtype), "steps": 28, "baseline_cfg": sargs.baseline_cfg,
        "seed": a.seed, "rounds": a.rounds, "k": a.k, "guard_thresh": a.guard_thresh, "operators": OPERATORS,
        "vlm": json.loads(urllib.request.urlopen(a.vlm_server + "/health", timeout=10).read()), "jev": json.loads(urllib.request.urlopen(a.jev_server + "/health", timeout=10).read()),
        "output_rule": "round with most `present` checklist elements, ties -> latest; no reward used in the loop", "readout": "v2: observe-then-grade VLM + per-element Jev yes/no with image; present requires both", "proposer": "v5: statuses only, original-vocabulary rule (soft), lexical RETENTION filter before the Jev guard; additions judged by Jev; sampled retries", "reward_for_export": rm.describe()}, indent=1))
    with torch.no_grad(), recp.open("a") as rec:
        for pid in todo:
            prompt = pmap[pid]; t0 = time.time(); ctx.nfe = 0; nv = nj = 0; tv = tj = 0.0
            try:
                elements, s_ = extract_checklist(a.vlm_server, prompt)
            except Exception as exc:
                print(f"[{a.arm}] {pid} SKIPPED: checklist unparsable after retries ({str(exc)[:90]})", flush=True); continue
            if not elements:
                print(f"[{a.arm}] {pid} SKIPPED: empty checklist", flush=True); continue
            nv += 1; tv += s_
            (run_dir / "checklists" / f"{pid}.json").write_text(json.dumps({"prompt": prompt, "elements": elements}, indent=1))
            print(f"[{a.arm}] {pid} checklist: " + " | ".join(f'{e["id"]}:{e["text"][:40]}' for e in elements), flush=True)
            wording = prompt; emb_cache = {}; rounds = []; stop_reason = "max_rounds"
            for r_ in range(a.rounds + 1):
                if wording not in emb_cache: emb_cache[wording] = su.encode_variants(ctx, [wording])
                img, ir = su.run_baseline(sargs, ctx, emb_cache[wording], rm, prompt, a.seed, cfg_scale=float(sargs.baseline_cfg))
                ip = run_dir / "rounds" / f"{pid}_r{r_}.png"; img.save(ip)
                statuses, s_, obs = readout(a.vlm_server, prompt, elements, img, jev=a.jev_server, seed=a.seed + 1000 * r_); nv += 1; nj += len(elements); tv += s_
                n_present = sum(1 for s in statuses if s["status"] == "present")
                rd = {"round": r_, "wording": wording, "image": str(ip), "ir": float(ir), "observations": obs, "statuses": statuses, "n_present": n_present, "n_elements": len(elements)}
                print(f"[{a.arm}] {pid} r{r_}: present {n_present}/{len(elements)} ir={ir:.3f} wording=“{wording[:90]}”", flush=True)
                if n_present == len(elements): rounds.append(rd); stop_reason = "all_present"; break
                if r_ == a.rounds: rounds.append(rd); break
                cands, s_ = propose(a.vlm_server, prompt, wording, elements, statuses, a.k); nv += 1; tv += s_
                for c in cands:
                    if not c.get("lexical_ok", True): c["guard_p_yes"], c["guard_spread"], c["passed_guard"] = None, None, False; continue
                    c["guard_p_yes"], c["guard_spread"], ms = guard(a.jev_server, prompt, c["wording"], seed=a.seed + r_); nj += 1; tj += ms / 1000
                    c["passed_guard"] = c["guard_p_yes"] >= a.guard_thresh
                tried = {d["wording"] for d in rounds} | {wording}
                for c in cands: c["already_tried"] = c["wording"] in tried
                surv = [c for c in cands if c["passed_guard"] and not c["already_tried"]]; rd["candidates"] = cands   # no-repeat: never re-sample a wording already graded
                if not surv: rounds.append(rd); stop_reason = "no_candidate_passed_guard"; break
                idx, probs, spread, ms = decide(a.jev_server, prompt, wording, elements, statuses, surv, img, seed=a.seed + 100 + r_); nj += 1; tj += ms / 1000
                rd["decision"] = {"options": [c["wording"] for c in surv] + ["KEEP"], "probs": probs, "spread": spread, "chosen_index": idx}
                rounds.append(rd)
                if idx == len(surv): stop_reason = "jev_keep"; break
                wording = surv[idx]["wording"]
            best = max(rounds, key=lambda d: (d["n_present"], d["round"]))
            out = run_dir / f"{pid}_{a.arm}.png"; Path(best["image"]).read_bytes() and out.write_bytes(Path(best["image"]).read_bytes())
            rec.write(json.dumps({"prompt_id": pid, "prompt": prompt, "arm": a.arm, "seed": a.seed, "elements": elements, "rounds": rounds, "stop_reason": stop_reason,
                                  "final_round": best["round"], "final_wording": best["wording"], "final_present": f'{best["n_present"]}/{len(elements)}',
                                  "search_score": float(best["ir"]), "image": str(out), "image_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                  "nfe_transformer": int(ctx.nfe), "n_vlm_calls": nv, "vlm_seconds": round(tv, 1), "n_jev_calls": nj, "jev_seconds": round(tj, 1),
                                  "seconds": round(time.time() - t0, 1)}) + "\n"); rec.flush()
            print(f"[{a.arm}] {pid} done: {stop_reason}, output round {best['round']} ({best['n_present']}/{len(elements)} present) ir={best['ir']:.3f} nfe={ctx.nfe} vlm={nv}/{tv:.0f}s jev={nj}/{tj:.0f}s {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
