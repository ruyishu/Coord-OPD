import os
"""Comprehensive, reviewer-aligned statistical summary:
1) All 4-seed completed arms + strict same-seed matched tables (for 2, 3, or 4 completed seeds of HS/AT/EF)
2) Primary metric: Official DocVQA ANLS; Secondary: Normalized EM; Reference: Substring
3) Same-seed aligned boxtest (miss, cov80, mIoU, IoU>=0.5, area/GT) & rollout localization (1st_miss, all_miss, cov80, crops, multi)
4) Conditional reading accuracy in ANLS & EM: both on own cov>=0.8 subset (with denominator) AND on the common intersection of cov>=0.8 questions
5) Per-seed paired differences (WAT - WH, WAT - HS, WAT - WA, WEF - WH) with page-clustered (group_id) standard errors
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

WS = Path(os.environ.get('COORD_OPD_WORKSPACE', str(Path(__file__).resolve().parent)))
BASE = WS / 'probe10_20261028'
PE = WS / 'coord_opd_results/eval/per_example'
BT = BASE / 'boxtest'
SEEDS = (42, 17, 2026, 7)
ARMS = ('A', 'WH', 'HS', 'HS5', 'ATG', 'AT', 'EF', 'TK1', 'TK2', 'TK3', 'TKN', 'TKV2')
NB = {'docvqa_dev': 125, 'docvqa_heldout': 279}

GT_ANS = {}
for b in NB:
    p = BASE / 'lowres/data' / f'{b}.jsonl'
    for l in p.read_text(encoding='utf-8').splitlines():
        if l.strip():
            r = json.loads(l)
            ans = r.get('answers') or r.get('possible_answers') or ([r['answer']] if 'answer' in r else [])
            if isinstance(ans, str):
                ans = [ans]
            GT_ANS[(b, r['id'])] = [str(a) for a in ans]

states = [json.loads(l) for k in range(2) for l in (BT / f'states_shard{k}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
ev_states = [s for s in states if s['split'].startswith('eval') and s.get('gt')]


def norm_em(s: str) -> str:
    s = re.sub(r'\s+', ' ', (s or '').strip().lower())
    return s.strip(' .,;:!?"\'`()[]{}')


def edit_dist(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def anls_single(pred: str, gts: list) -> float:
    p = ' '.join((pred or '').strip().lower().split())
    best = 0.0
    for g in gts:
        g_clean = ' '.join(str(g).strip().lower().split())
        if not g_clean and not p:
            return 1.0
        if not g_clean or not p:
            continue
        d = edit_dist(p, g_clean)
        nl = d / max(len(p), len(g_clean))
        sim = 1.0 - nl if nl < 0.5 else 0.0
        if sim > best:
            best = sim
    return best


def strict_em(pred: str, gts: list) -> float:
    p = norm_em(pred)
    if not p:
        return 0.0
    return 1.0 if any(p == norm_em(g) for g in gts) else 0.0


def load_run(prefix):
    out = {}
    for b, n in NB.items():
        hits = sorted(PE.glob(f'{prefix}_step*__{b}__1mp_ov262144.jsonl'))
        if not hits:
            return None
        d_b = {}
        for l in hits[-1].read_text(encoding='utf-8').splitlines():
            if l.strip():
                r = json.loads(l)
                pred = r.get('pred_answer', r.get('final_answer', r.get('prediction', '')))
                gts = GT_ANS.get((b, r['id'])) or r.get('answers') or r.get('gt_answers') or ([r['gt_answer']] if 'gt_answer' in r else [])
                r['_anls'] = anls_single(pred, gts)
                r['_em'] = strict_em(pred, gts)
                r['_sub'] = float(r['correct'])
                d_b[r['id']] = r
        if len(d_b) != n:
            return None
        for k, v in d_b.items():
            out[(b, k)] = v
    return out if len(out) == 404 else None


def load_boxtest(tag):
    p = BT / f'measure_{tag}.json'
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding='utf-8'))
    bmap = m.get('boxes', {})
    i25, i30, i50 = 0, 0, 0
    for st in ev_states:
        k = st['split'] + '|' + st['id']
        q = bmap.get(k)
        if q:
            b = [v / 1000.0 for v in q]
            g = st['gt']
            iw = max(0.0, min(b[2], g[2]) - max(b[0], g[0]))
            ih = max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
            inter = iw * ih
            u = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) + max(0.0, g[2] - g[0]) * max(0.0, g[3] - g[1]) - inter
            v = inter / u if u > 0 else 0.0
            if v >= 0.25:
                i25 += 1
            if v >= 0.30:
                i30 += 1
            if v >= 0.50:
                i50 += 1
    m['iou25'] = 100.0 * i25 / len(ev_states)
    m['iou30'] = 100.0 * i30 / len(ev_states)
    m['iou50'] = 100.0 * i50 / len(ev_states)
    return m


def msd(xs):
    if not xs:
        return float('nan'), float('nan')
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


RUNS = {}
BOXES = {}
for a in ARMS:
    RUNS[a] = {}
    BOXES[a] = {}
    for s in SEEDS:
        rid = f'W{a}_lr1e4_s{s}'
        d = load_run(rid)
        if d:
            RUNS[a][s] = d
        bm = load_boxtest(rid)
        if bm:
            BOXES[a][s] = bm

print('=== 1. Overall Per-Seed Table (404 questions: ANLS / Normalized EM / Substring) ===', flush=True)
print(f"{'arm':6s} | {'n':>1s} | {'Official ANLS':16s} | {'Normalized EM':16s} | {'Substring (old)':16s} | {'ANLS per seed (42 / 17 / 2026 / 7)':34s} | {'EM per seed (42 / 17 / 2026 / 7)'}", flush=True)
for a in ARMS:
    if not RUNS[a]:
        continue
    anls_l, em_l, sub_l = [], [], []
    anls_str, em_str = [], []
    for s in SEEDS:
        if s in RUNS[a]:
            d = RUNS[a][s]
            av = 100.0 * sum(r['_anls'] for r in d.values()) / 404.0
            ev = 100.0 * sum(r['_em'] for r in d.values()) / 404.0
            sv = 100.0 * sum(r['_sub'] for r in d.values()) / 404.0
            anls_l.append(av)
            em_l.append(ev)
            sub_l.append(sv)
            anls_str.append(f'{av:5.2f}')
            em_str.append(f'{ev:5.2f}')
        else:
            anls_str.append('  -  ')
            em_str.append('  -  ')
    ma, sda = msd(anls_l)
    me, sde = msd(em_l)
    ms, sds = msd(sub_l)
    sd_fmt_a = f'{sda:4.2f}' if len(anls_l) > 1 else ' N/A'
    sd_fmt_e = f'{sde:4.2f}' if len(em_l) > 1 else ' N/A'
    sd_fmt_s = f'{sds:4.2f}' if len(sub_l) > 1 else ' N/A'
    print(f"W{a:5s} | {len(anls_l):1d} | {ma:6.2f}% +-{sd_fmt_a}% | {me:6.2f}% +-{sd_fmt_e}% | {ms:6.2f}% +-{sd_fmt_s}% | "
          + ' / '.join(anls_str) + ' | ' + ' / '.join(em_str), flush=True)

# Find common completed seeds for WAT
wat_seeds = [s for s in SEEDS if s in RUNS['AT'] and s in BOXES['AT']]
if wat_seeds:
    print(f'\n=== 2. Strict Same-Seed Aligned Table on WAT Completed Seeds {tuple(wat_seeds)} (QA + Boxtest + Rollout + Conditional Reading) ===', flush=True)
    print(f"{'arm':6s} | {'ANLS':>6s} {'EM':>6s} {'Sub':>6s} | {'bt_miss':>7s} {'bt_cov80':>8s} {'bt_mIoU':>7s} {'IoU>=.5':>7s} | "
          f"{'1st_ms':>6s} {'all_ms':>6s} {'rl_cov80':>8s} {'crops':>5s} {'multi':>6s} | {'ANLS|cov80 (n)':>16s} {'EM|cov80':>9s}", flush=True)
    # Also find common intersection of (seed, question) on dev where WA, WH, HS, AT all have cov>=0.8
    common_cov80_keys = set()
    for s in wat_seeds:
        for qid in range(10000):
            pass
    for a in ('A', 'WH', 'HS', 'HS5', 'ATG', 'AT', 'EF', 'TKN', 'TKV2'):
        if not all(s in RUNS[a] and s in BOXES[a] for s in wat_seeds):
            continue
        anls_v = sum(sum(r['_anls'] for r in RUNS[a][s].values()) / 404.0 for s in wat_seeds) * 100.0 / len(wat_seeds)
        em_v = sum(sum(r['_em'] for r in RUNS[a][s].values()) / 404.0 for s in wat_seeds) * 100.0 / len(wat_seeds)
        sub_v = sum(sum(r['_sub'] for r in RUNS[a][s].values()) / 404.0 for s in wat_seeds) * 100.0 / len(wat_seeds)
        bt_miss = sum(BOXES[a][s]['miss'] for s in wat_seeds) / len(wat_seeds)
        bt_cov = sum(BOXES[a][s]['cover80'] for s in wat_seeds) / len(wat_seeds)
        bt_iou = sum(BOXES[a][s]['iou_gt'] for s in wat_seeds) / len(wat_seeds)
        bt_i50 = sum(BOXES[a][s]['iou50'] for s in wat_seeds) / len(wat_seeds)

        fmiss, amiss, rcov, crops_l, multi_l = [], [], [], [], []
        c80_anls, c80_em = [], []
        for s in wat_seeds:
            dev_rows = [r for (b, _), r in RUNS[a][s].items() if b == 'docvqa_dev']
            all_rows = list(RUNS[a][s].values())
            for r in dev_rows:
                fc = r['localization']['first_crop']['mean_coverage']
                ac = r['localization']['all_crops']['mean_coverage']
                fmiss.append(100.0 if fc <= 1e-9 else 0.0)
                amiss.append(100.0 if ac <= 1e-9 else 0.0)
                rcov.append(100.0 if ac >= 0.8 else 0.0)
                if ac >= 0.8:
                    c80_anls.append(100.0 * r['_anls'])
                    c80_em.append(100.0 * r['_em'])
            for r in all_rows:
                crops_l.append(r['crop_count'])
                multi_l.append(100.0 if r['crop_count'] > 1 else 0.0)
        print(f"W{a:5s} | {anls_v:5.2f}% {em_v:5.2f}% {sub_v:5.2f}% |  {bt_miss:5.1f}%   {bt_cov:5.1f}%   {bt_iou:5.3f}   {bt_i50:5.1f}% | "
              f"{sum(fmiss)/len(fmiss):5.1f}% {sum(amiss)/len(amiss):5.1f}%   {sum(rcov)/len(rcov):5.1f}%  {sum(crops_l)/len(crops_l):4.2f}  {sum(multi_l)/len(multi_l):5.1f}% | "
              f"{sum(c80_anls)/max(1,len(c80_anls)):5.2f}% ({len(c80_anls):3d}/{len(fmiss):3d})   {sum(c80_em)/max(1,len(c80_em)):5.2f}%", flush=True)

    # Common cov>=0.8 intersection across WA, WH, HS, AT on wat_seeds
    if all(all(s in RUNS[a] for s in wat_seeds) for a in ('A', 'WH', 'HS', 'AT')):
        inter_keys = []
        for s in wat_seeds:
            for (b, qid), r in RUNS['A'][s].items():
                if b == 'docvqa_dev':
                    if all(RUNS[a][s][(b, qid)]['localization']['all_crops']['mean_coverage'] >= 0.8 for a in ('A', 'WH', 'HS', 'AT')):
                        inter_keys.append((s, b, qid))
        print(f'\n=== 3. Fixed Common-Intersection Reading Check (questions where WA, WH, HS, AT ALL achieved cov>=0.8, n={len(inter_keys)}) ===', flush=True)
        for a in ('A', 'WH', 'HS', 'AT'):
            ia = sum(100.0 * RUNS[a][s][(b, qid)]['_anls'] for (s, b, qid) in inter_keys) / max(1, len(inter_keys))
            ie = sum(100.0 * RUNS[a][s][(b, qid)]['_em'] for (s, b, qid) in inter_keys) / max(1, len(inter_keys))
            isub = sum(100.0 * RUNS[a][s][(b, qid)]['_sub'] for (s, b, qid) in inter_keys) / max(1, len(inter_keys))
            print(f"  W{a:4s} on identical cov>=0.8 subset (n={len(inter_keys)}): ANLS={ia:5.2f}% | EM={ie:5.2f}% | Sub={isub:5.2f}%", flush=True)

print('\n=== 4. Per-Seed Paired Differences (WAT vs WH / HS / WA, and WEF vs WH / WA) with Page-Clustered SE ===', flush=True)


def page_clustered_se(diffs_by_page):
    # diffs_by_page: dict[page_id -> list of item diffs in pct pts]
    G = len(diffs_by_page)
    N = sum(len(v) for v in diffs_by_page.values())
    if G <= 1 or N == 0:
        return 0.0
    all_mean = sum(sum(v) for v in diffs_by_page.values()) / N
    # Cluster residual sum of squares
    ss = sum((sum(d - all_mean for d in v)) ** 2 for v in diffs_by_page.values())
    return (G / (G - 1) * ss) ** 0.5 / N


for cand in ('AT', 'ATG', 'HS5', 'HS', 'EF', 'TKV2'):
    if not RUNS[cand]:
        continue
    for ref in ('WH', 'HS', 'HS5', 'ATG', 'A'):
        if cand == ref:
            continue
        common_s = [s for s in SEEDS if s in RUNS[cand] and s in RUNS[ref]]
        if not common_s:
            continue
        for metric_key, metric_name in (('_anls', 'ANLS'), ('_em', 'EM')):
            s_diffs = []
            cells = []
            pooled_by_page = defaultdict(list)
            for s in common_s:
                x, y = RUNS[cand][s], RUNS[ref][s]
                by_p = defaultdict(list)
                for k in x:
                    diff_pt = 100.0 * (x[k][metric_key] - y[k][metric_key])
                    pg = f"{k[0]}|{x[k].get('group_id') or k[1]}"
                    by_p[pg].append(diff_pt)
                    pooled_by_page[pg].append(diff_pt / len(common_s))
                d_mean = sum(sum(v) for v in by_p.values()) / 404.0
                d_se = page_clustered_se(by_p)
                s_diffs.append(d_mean)
                cells.append(f"s{s}:{d_mean:+5.2f}(+-{d_se:4.2f})")
            m_d, sd_d = msd(s_diffs)
            p_se = page_clustered_se(pooled_by_page)
            pos = sum(1 for d in s_diffs if d > 0)
            print(f"  W{cand:4s}-W{ref:2s} [{metric_name:4s}] " + '  '.join(cells) +
                  f" | seed_mean={m_d:+5.2f} (seed_sd={sd_d:4.2f}, page_SE={p_se:4.2f}), pos_seeds={pos}/{len(common_s)}", flush=True)
