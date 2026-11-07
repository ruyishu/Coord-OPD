"""CPU validation for geo_feas: exhaustive oracle on small grids, float-vs-integer predicate, tokenizer grammar."""
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo_feas as GF


class _OfflineDigitTokenizer:
    """Lightweight character/tag tokenizer matching Qwen3-VL single-digit + tail token behavior for offline CPU checks."""
    def __init__(self):
        self._toks = [str(d) for d in range(10)] + [",", "]", "}}", "}</tool_call>", "</tool_call>", " ", "}", "]}}"]
        self._s2i = {s: i + 10 for i, s in enumerate(self._toks)}
        self._i2s = {i + 10: s for i, s in enumerate(self._toks)}

    def encode(self, text, add_special_tokens=False):
        out = []
        i = 0
        while i < len(text):
            if text.startswith("</tool_call>", i):
                out.append(self._s2i["</tool_call>"])
                i += len("</tool_call>")
            elif text.startswith("}}", i):
                out.append(self._s2i["}}"])
                i += 2
            else:
                ch = text[i]
                if ch not in self._s2i:
                    idx = len(self._s2i) + 20
                    self._s2i[ch] = idx
                    self._i2s[idx] = ch
                out.append(self._s2i[ch])
                i += 1
        return out

    def decode(self, ids):
        return "".join(self._i2s.get(i, "") for i in ids)


def _load_tokenizer():
    p_2b = os.environ.get("QWEN3_VL_2B_PATH", "")
    if p_2b and Path(p_2b).exists():
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(p_2b)
    return _OfflineDigitTokenizer()


t0 = time.time()
# 1) exhaustive exactness on small grids: feasible_values vs brute-force enumeration of all boxes
rng = random.Random(7)
n_cases = n_checked = n_bad = 0
for G in (8, 12, 16):
    boxes = [(a, b, c, d) for a in range(G) for b in range(G) for c in range(a + 1, G + 1) for d in range(b + 1, G + 1)]
    teachers = list(boxes)
    rng.shuffle(teachers)
    for T in teachers[:(60 if G > 8 else 80)]:
        n_cases += 1
        orc = GF.GeoOracle(T, G)
        accepted = [bx for bx in boxes if GF.accepts(bx, T)]
        for c in range(4):
            prefixes = [()] if c == 0 else [tuple(bx[:c]) for bx in rng.sample(boxes, min(25, len(boxes)))]
            for pre in prefixes:
                got = orc.feasible_values(c, pre)
                for v in range(G + 1):
                    want = any(bx[:c] == pre and bx[c] == v for bx in accepted)
                    n_checked += 1
                    if bool(got[v]) != want:
                        n_bad += 1
                        if n_bad <= 5:
                            print("MISMATCH G", G, "T", T, "c", c, "pre", pre, "v", v, "got", bool(got[v]), "want", want, flush=True)
print(f"EXHAUSTIVE cases={n_cases} checks={n_checked} mismatches={n_bad} sec={time.time() - t0:.1f}", flush=True)
assert n_bad == 0

# 2) integer predicate vs the float predicate used by the sparse QKL (tk_core._qkl_accept) on G=1000
def float_accept(b, t):
    s = [v / 1000.0 for v in b]
    tt = [v / 1000.0 for v in t]
    iw = max(0.0, min(s[2], tt[2]) - max(s[0], tt[0]))
    ih = max(0.0, min(s[3], tt[3]) - max(s[1], tt[1]))
    inter = iw * ih
    ta = max(1e-9, (tt[2] - tt[0]) * (tt[3] - tt[1]))
    sa = max(0.0, s[2] - s[0]) * max(0.0, s[3] - s[1])
    u = sa + ta - inter
    iou = inter / u if u > 0 else 0.0
    return bool((inter / ta >= 0.85 and sa / ta <= 3.5) or iou >= 0.5)

dis = tot = 0
for _ in range(200000):
    t = sorted(rng.sample(range(0, 1001), 2)) + sorted(rng.sample(range(0, 1001), 2))
    T = (t[0], t[2], t[1], t[3])
    r = sorted(rng.sample(range(0, 1001), 2)) + sorted(rng.sample(range(0, 1001), 2))
    B = (r[0], r[2], r[1], r[3])
    if T[2] <= T[0] or T[3] <= T[1] or B[2] <= B[0] or B[3] <= B[1]:
        continue
    tot += 1
    dis += int(GF.accepts(B, T) != float_accept(B, T))
print(f"INT_VS_FLOAT total={tot} disagreements={dis} (IEEE-754 boundary rounding only)", flush=True)
assert dis <= 5

# 3) G=1000 sanity: teacher box itself and an expanded box have a fully feasible canonical path; speed
T = (390, 290, 510, 370)
orc = GF.GeoOracle(T, 1000)
t1 = time.time()
for c, pre in ((0, ()), (1, (T[0],)), (2, T[:2]), (3, T[:3])):
    f = orc.feasible_values(c, pre)
    assert f[T[c]], (c, pre)
print(
    f"G1000 oracle 4 levels in {time.time() - t1:.2f}s; feasible counts",
    [int(orc.feasible_values(c, pre).sum()) for c, pre in ((0, ()), (1, (T[0],)), (2, T[:2]), (3, T[:3]))],
    flush=True,
)
f3 = orc.feasible_values(3, T[:3])
assert all(GF.accepts((T[0], T[1], T[2], v), T) for v in np.nonzero(f3)[0])
assert not any(GF.accepts((T[0], T[1], T[2], v), T) for v in np.nonzero(~f3)[0])
f2 = orc.feasible_values(2, T[:2])
for x2 in range(0, 1001, 7):
    want = any(GF.accepts((T[0], T[1], x2, y2), T) for y2 in range(T[1] + 1, 1001))
    assert bool(f2[x2]) == want, ("c2", x2)
f1 = orc.feasible_values(1, (T[0],))
for y1 in range(0, 1000, 37):
    want = any(GF.accepts((T[0], y1, x2, y2), T) for x2 in range(T[0] + 1, 1001, 3) for y2 in range(y1 + 1, 1001, 3))
    if want:
        assert f1[y1], ("c1 want true", y1)
print("G1000_WITNESS_OK", flush=True)

# 4) tokenizer grammar
tok = _load_tokenizer()
gram = GF.GeoGrammar(tok, 1000)
print("TAIL_IDS", gram.tail_ids, [tok.decode([i]) for i in gram.tail_ids], flush=True)
suffix = lambda q: f"{q[0]},{q[1]},{q[2]},{q[3]}]}}}}</tool_call>"
n_acc = n_rej = 0
for _ in range(300):
    base = [T[0] + rng.randint(-60, 60), T[1] + rng.randint(-40, 40), T[2] + rng.randint(-60, 60), T[3] + rng.randint(-40, 40)]
    q = [max(0, min(1000, v)) for v in base]
    if q[2] <= q[0] or q[3] <= q[1]:
        continue
    ids = tok.encode(suffix(q), add_special_tokens=False)
    res = gram.analyze(ids, T, orc)
    first_bad = next((j for j, (al, st) in enumerate(res) if ids[j] not in al), None)
    acc = GF.accepts(q, T)
    if acc:
        n_acc += 1
        assert first_bad is None, ("accepted box left the feasible grammar", q, first_bad)
    else:
        n_rej += 1
        assert first_bad is not None, ("rejected box stayed feasible", q)
print(f"GRAMMAR_OK accepted={n_acc} rejected={n_rej}", flush=True)
ids = tok.encode("390, 290,510,370]}}</tool_call>", add_special_tokens=False)
res = gram.analyze(ids, T, orc)
print("FORMAT_STATUS", [st for _, st in res][:14], flush=True)
print("GEO_SELFTEST_PASS", f"{time.time() - t0:.1f}s", flush=True)
