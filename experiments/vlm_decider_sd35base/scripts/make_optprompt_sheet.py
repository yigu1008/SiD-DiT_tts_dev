#!/usr/bin/env python3
"""Eye-check sheets for the deliberate prompt optimizer: one PNG per prompt with every round's image + wording, checklist statuses,
candidates (operator, guard P(yes)) and Jev's choice.  python make_optprompt_sheet.py [run_dir=probe/opt_prompt]"""
import json, sys, os, textwrap
from PIL import Image, ImageDraw, ImageFont
R = "/data/ygu/runs/vlm_decider_sd35base_2026-10-16"; D = sys.argv[1] if len(sys.argv) > 1 else f"{R}/probe/opt_prompt"
try: F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15); FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except Exception: F = FB = ImageFont.load_default()
os.makedirs(f"{D}/sheets", exist_ok=True); W, TW = 384, 620
for line in open(f"{D}/records.jsonl"):
    r = json.loads(line); rounds = r["rounds"]; H = max(1, len(rounds)) * (W + 8) + 60
    sheet = Image.new("RGB", (W + TW + 24, H), "white"); d = ImageDraw.Draw(sheet)
    d.text((8, 6), f'{r["prompt_id"]}  ORIGINAL: {r["prompt"]}', fill="black", font=FB)
    d.text((8, 28), "checklist: " + " | ".join(f'{e["id"]}:{e["text"]}' for e in r["elements"]), fill=(60, 60, 60), font=F)
    y = 56
    for rd in rounds:
        im = Image.open(rd["image"]).convert("RGB").resize((W, W)); sheet.paste(im, (8, y))
        mark = " <== OUTPUT" if rd["round"] == r["final_round"] else ""
        lines = [f'round {rd["round"]}  present {rd["n_present"]}/{rd["n_elements"]}  IR {rd["ir"]:.3f}{mark}', "wording: " + rd["wording"], ""]
        for s in rd["statuses"]: lines.append(f'  {s["id"]} {s["status"]:8s} {s["note"]}')
        if rd.get("candidates"):
            lines.append(""); lines.append("candidates:")
            for i, c in enumerate(rd["candidates"]): lines.append(f'  [{c["operator"]}] guard={(c.get("guard_p_yes") if c.get("guard_p_yes") is not None else float("nan")):.2f}{"" if c.get("passed_guard") else " X"}: {c["wording"]}')
        if rd.get("decision"):
            dec = rd["decision"]; lines.append(f'Jev: ' + ", ".join(f"{p:.2f}" for p in dec["probs"]) + f'  -> {"KEEP" if dec["chosen_index"] == len(dec["options"]) - 1 else "option " + str(dec["chosen_index"] + 1)}')
        ty = y
        for ln in lines:
            for wl in textwrap.wrap(ln, 78) or [""]:
                d.text((W + 16, ty), wl, fill="black" if not wl.startswith("  ") else (40, 40, 40), font=FB if ln.startswith("round") else F); ty += 18
                if ty > y + W - 10: break
            if ty > y + W - 10: d.text((W + 16, ty), "…", fill="gray", font=F); break
        y += W + 8
    out = f'{D}/sheets/{r["prompt_id"]}_optprompt.png'; sheet.save(out); print("wrote", out, f'stop={r["stop_reason"]} output round {r["final_round"]} {r["final_present"]}')
