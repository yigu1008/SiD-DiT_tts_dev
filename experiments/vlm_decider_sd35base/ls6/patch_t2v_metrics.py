"""Make every model-backend import in t2v_metrics optional except CLIP-FlanT5 (idempotent per file).
We only use VQAScore(clip-flant5-xxl); the other backends pull external repos / flash_attn / LLaVA-NeXT at import time.
A failed backend import becomes an empty *_MODELS list + None class, so list/get of clip-flant5-xxl still works."""
import re, sys, sysconfig, os
base = os.path.join(sysconfig.get_paths()["purelib"], "t2v_metrics", "models")
for fam in ("vqascore_models", "clipscore_models", "itmscore_models"):
    init = os.path.join(base, fam, "__init__.py")
    if not os.path.exists(init): continue
    src = open(init).read()
    if "# (patched-optional-backends)" in src: print("already patched:", fam); continue
    out = []
    for line in src.splitlines():
        m = re.match(r"^from \.(\w+) import ([\w, ]+)$", line)
        if m and m.group(1) != "clip_t5_model":
            mod = m.group(1); names = [n.strip() for n in m.group(2).split(",")]
            fallback = "; ".join(f"{n} = []" if n.endswith("_MODELS") else f"{n} = None" for n in names)
            out.append(f"try:\n    {line}\nexcept Exception:  # (patched-optional-backends)\n    {fallback}")
        else: out.append(line)
    open(init, "w").write("\n".join(out) + "\n"); print("patched:", fam)
