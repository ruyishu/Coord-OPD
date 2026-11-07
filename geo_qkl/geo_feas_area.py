"""Exact geometric prefix-feasibility for Geo-QKL (CPU, numpy only).

Acceptable set for a teacher box T=(tx1,ty1,tx2,ty2) on the integer grid 0..G (G=1000):
  b is accepted  iff  (cov(b,T) >= 0.85 and area(b)/area(T) <= 3.5)  or  IoU(b,T) >= 0.5,
evaluated with exact integer arithmetic. Boxes must satisfy 0 <= x1 < x2 <= G and 0 <= y1 < y2 <= G.

feasible_values(c, fixed) returns a boolean array over 0..G: value v is True iff some completion of the
remaining coordinates (after fixing the first c coordinates to `fixed`) yields an accepted box.
The y-axis best response is closed-form and the x2 free axis is reduced to non-dominated values
(x2 <= tx2), so every True/False is exact, not a search outcome.
"""
import numpy as np

G_DEFAULT = 1000

import os as _os
# area cap as a rational AREA_P/AREA_Q (7/2 = 3.5 is the frozen default); module globals, read at call time
AREA_P = int(_os.environ.get("GEO_AREA_P", "7"))
AREA_Q = int(_os.environ.get("GEO_AREA_Q", "2"))


def accepts(box, teacher):
    """Exact integer acceptance predicate."""
    x1, y1, x2, y2 = [int(v) for v in box]
    tx1, ty1, tx2, ty2 = [int(v) for v in teacher]
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        return False
    iw = max(0, min(x2, tx2) - max(x1, tx1))
    ih = max(0, min(y2, ty2) - max(y1, ty1))
    inter = iw * ih
    ta = (tx2 - tx1) * (ty2 - ty1)
    a = (x2 - x1) * (y2 - y1)
    if 100 * inter >= 85 * ta and AREA_Q * a <= AREA_P * ta:
        return True
    return 3 * inter >= a + ta


class GeoOracle:
    def __init__(self, teacher, G=G_DEFAULT):
        self.G = int(G)
        self.t = [int(v) for v in teacher]
        tx1, ty1, tx2, ty2 = self.t
        if not (0 <= tx1 < tx2 <= self.G and 0 <= ty1 < ty2 <= self.G):
            raise ValueError(f'invalid teacher box {teacher}')
        self.tw = tx2 - tx1
        self.th = ty2 - ty1
        self.ta = self.tw * self.th
        self._cache = {}

    # ---- y-axis best responses (arrays over broadcastable iw, w) -------------------------------------
    def _y_free_both(self, iw, w):
        """Both y1,y2 free (y1<y2 in 0..G). True where some (ih,h) satisfies a condition. Exact."""
        th, ta = self.th, self.ta
        iw = np.asarray(iw, dtype=np.int64)
        w = np.asarray(w, dtype=np.int64)
        ok = np.zeros(np.broadcast(iw, w).shape, dtype=bool)
        pos = iw >= 1
        # cond1: choose h = k = ih in [1, th]; need 100*iw*k >= 85*ta, 2*w*k <= 7*ta
        a_need = -(-(85 * ta) // (100 * np.maximum(iw, 1)))  # ceil(85 ta / (100 iw)) = kmin1
        kmin1 = np.maximum(a_need, 1)
        kmax1 = np.minimum(th, (AREA_P * ta) // (AREA_Q * np.maximum(w, 1)))
        ok |= pos & (kmin1 <= kmax1)
        # cond2: IoU>=1/2 <=> 3*inter >= a + ta; with h = k: k*(3*iw - w) >= ta ; best k = th
        ok |= pos & (th * (3 * iw - w) >= ta)
        return ok

    def _y_fixed_low(self, y1, iw, w):
        """y1 fixed (array), y2 free in (y1, G]. Exact (non-dominated y2 <= ty2)."""
        th, ta, ty1, ty2 = self.th, self.ta, self.t[1], self.t[3]
        y1 = np.asarray(y1, dtype=np.int64)
        iw = np.asarray(iw, dtype=np.int64)
        w = np.asarray(w, dtype=np.int64)
        m = np.maximum(y1, ty1)
        ok = np.zeros(np.broadcast(y1, iw, w).shape, dtype=bool)
        valid = (y1 < ty2) & (iw >= 1)
        kcap = ty2 - m  # max intersection height, attained with y2 = ty2
        # cond1: h = (m - y1) + k, k in [1, kcap]
        kmin1 = np.maximum(-(-(85 * ta) // (100 * np.maximum(iw, 1))), 1)
        base = m - y1
        # 2*w*(base + k) <= 7 ta  ->  k <= floor(7 ta / (2 w)) - base
        kmax1 = np.minimum(kcap, (AREA_P * ta) // (AREA_Q * np.maximum(w, 1)) - base)
        ok |= valid & (kmin1 <= kmax1)
        # cond2: 3*iw*k >= w*(base + k) + ta  ->  k*(3*iw - w) >= ta + w*base ; best k = kcap
        ok |= valid & (kcap * (3 * iw - w) >= ta + w * base)
        return ok

    def _y_fixed_both(self, y1, y2, iw, w):
        ty1, ty2, ta = self.t[1], self.t[3], self.ta
        y1 = np.asarray(y1, dtype=np.int64)
        y2 = np.asarray(y2, dtype=np.int64)
        iw = np.asarray(iw, dtype=np.int64)
        w = np.asarray(w, dtype=np.int64)
        ih = np.maximum(0, np.minimum(y2, ty2) - np.maximum(y1, ty1))
        h = y2 - y1
        inter = iw * ih
        a = w * h
        c1 = (100 * inter >= 85 * ta) & (AREA_Q * a <= AREA_P * ta)
        c2 = 3 * inter >= a + ta
        return (y1 < y2) & (w >= 1) & (c1 | c2)

    # ---- value feasibility per coordinate ---------------------------------------------------------------
    def feasible_values(self, c, fixed):
        key = (c, tuple(int(v) for v in fixed))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        G = self.G
        tx1, ty1, tx2, ty2 = self.t
        v = np.arange(G + 1, dtype=np.int64)
        out = np.zeros(G + 1, dtype=bool)
        if c == 0:  # choose x1; x2,y1,y2 free
            for x1 in range(0, min(tx2, G + 1)):
                x2 = np.arange(x1 + 1, tx2 + 1, dtype=np.int64)  # x2 > tx2 is dominated
                if x2.size == 0:
                    continue
                iw = np.maximum(0, np.minimum(x2, tx2) - max(x1, tx1))
                w = x2 - x1
                if bool(self._y_free_both(iw, w).any()):
                    out[x1] = True
        elif c == 1:  # choose y1 given x1; x2,y2 free
            x1 = int(fixed[0])
            if x1 < tx2:
                x2 = np.arange(x1 + 1, tx2 + 1, dtype=np.int64)
                if x2.size:
                    iw = np.maximum(0, np.minimum(x2, tx2) - max(x1, tx1))
                    w = x2 - x1
                    y1 = v[:, None]
                    out = self._y_fixed_low(y1, iw[None, :], w[None, :]).any(axis=1)
        elif c == 2:  # choose x2 given x1,y1; y2 free
            x1, y1 = int(fixed[0]), int(fixed[1])
            x2 = v
            iw = np.maximum(0, np.minimum(x2, tx2) - max(x1, tx1))
            w = x2 - x1
            res = self._y_fixed_low(np.int64(y1), iw, w)
            out = res & (x2 > x1)
        elif c == 3:  # choose y2 given x1,y1,x2
            x1, y1, x2 = int(fixed[0]), int(fixed[1]), int(fixed[2])
            iw = max(0, min(x2, tx2) - max(x1, tx1))
            w = x2 - x1
            out = self._y_fixed_both(np.int64(y1), v, np.int64(iw), np.int64(w)) & (x2 > x1)
        else:
            raise ValueError(c)
        self._cache[key] = out
        return out


def prefix_sums(arr):
    return np.concatenate([[0], np.cumsum(arr.astype(np.int64))])


def values_with_prefix(feas, digits):
    """True iff some feasible value's canonical decimal string starts with `digits` (no leading zeros except '0')."""
    if digits == '':
        return bool(feas.any())
    if len(digits) > 1 and digits[0] == '0':
        return False
    p = int(digits)
    G = len(feas) - 1
    total = False
    for extra in range(0, 4):
        lo = p * (10 ** extra)
        hi = (p + 1) * (10 ** extra) - 1
        if lo > G:
            break
        hi = min(hi, G)
        if feas[lo:hi + 1].any():
            total = True
            break
        if p == 0:
            break
    return total


class GeoGrammar:
    """Token-level view of the canonical box suffix 'x1,y1,x2,y2]}}</tool_call>' for a given tokenizer."""

    def __init__(self, tokenizer, G=G_DEFAULT):
        self.tok = tokenizer
        self.G = G
        self.digit_id = {}
        for d in '0123456789':
            ids = tokenizer.encode(d, add_special_tokens=False)
            assert len(ids) == 1, (d, ids)
            self.digit_id[d] = ids[0]
        ids = tokenizer.encode(',', add_special_tokens=False)
        assert len(ids) == 1
        self.comma_id = ids[0]
        sample = tokenizer.encode('390,290,510,370]}}</tool_call>', add_special_tokens=False)
        n_prefix = 3 + 1 + 3 + 1 + 3 + 1 + 3
        assert sample[:n_prefix] == [self.digit_id[c] if c != ',' else self.comma_id for c in '390,290,510,370']
        self.tail_ids = sample[n_prefix:]
        self.id_digit = {v: k for k, v in self.digit_id.items()}

    def analyze(self, ids, teacher, oracle=None):
        """Walk the visited coordinate token ids. For each position j return (allowed_token_id_set, status).
        status: 'ok' (allowed non-empty), 'infeasible' (grammar fine but no accepted completion),
                'format' (prefix leaves the canonical grammar before/while still geometrically feasible)."""
        oracle = oracle or GeoOracle(teacher, self.G)
        vals, cur = [], ''
        in_tail, tail_pos = False, 0
        broken = None  # None | 'infeasible' | 'format'
        out = []
        for tid in ids:
            if broken is not None:
                out.append((set(), broken))
            elif in_tail:
                exp = self.tail_ids[tail_pos] if tail_pos < len(self.tail_ids) else None
                out.append(({exp}, 'ok') if exp is not None else (set(), 'format'))
            else:
                c = len(vals)
                feas = oracle.feasible_values(c, vals)
                allowed = set()
                for d, did in self.digit_id.items():
                    if values_with_prefix(feas, cur + d):
                        allowed.add(did)
                if cur != '' and (len(cur) == 1 or cur[0] != '0') and int(cur) <= self.G and feas[int(cur)]:
                    allowed.add(self.comma_id if c < 3 else self.tail_ids[0])
                out.append((allowed, 'ok' if allowed else 'infeasible'))
            # advance state with the visited token
            if broken is not None:
                continue
            if in_tail:
                if tail_pos < len(self.tail_ids) and tid == self.tail_ids[tail_pos]:
                    tail_pos += 1
                else:
                    broken = 'format'
                continue
            if tid in self.id_digit:
                nxt = cur + self.id_digit[tid]
                if len(nxt) > 1 and nxt[0] == '0':
                    broken = 'format'
                elif len(nxt) > 4 or int(nxt) > self.G:
                    broken = 'infeasible'
                else:
                    cur = nxt
                    if not values_with_prefix(oracle.feasible_values(len(vals), vals), cur):
                        broken = 'infeasible'
            elif tid == self.comma_id and cur != '' and len(vals) < 3:
                v = int(cur)
                if not oracle.feasible_values(len(vals), vals)[v]:
                    broken = 'infeasible'
                else:
                    vals.append(v)
                    cur = ''
            elif cur != '' and len(vals) == 3 and tid == self.tail_ids[0]:
                v = int(cur)
                if not oracle.feasible_values(3, vals)[v]:
                    broken = 'infeasible'
                else:
                    vals.append(v)
                    cur = ''
                    in_tail, tail_pos = True, 1
            else:
                broken = 'format'
        return out
