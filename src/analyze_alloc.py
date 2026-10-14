"""Token-allocation analysis on the X-Fi sweep: MICI labels, MICI router, baselines, curve.

Input: alloc_logits_*.npz from eval_alloc.py (logits R x N x 124 x 27 over (depth, mmwave, lidar)
token tiers {0,4,8,16,32}).  Router train/test halves are split by subject.
"""
import pilot_paths  # noqa: E402
import argparse
import itertools
import json
import os

import numpy as np
import torch

TIERS = [0, 4, 8, 16, 32]
STATES = list(itertools.product(TIERS, repeat=3))          # includes (0,0,0) = nothing seen yet
SID = {s: i for i, s in enumerate(STATES)}
NEXT = {0: 4, 4: 8, 8: 16, 16: 32}
MODS = ['depth', 'mmwave', 'lidar']
FULL = 96


def load(paths):
    """paths: sweep pieces covering consecutive sample ranges (same repeats); joined along samples."""
    ds = [np.load(p) for p in paths]
    L = np.concatenate([d['logits'].astype(np.float32) for d in ds], 1)   # R x N x 124 x 27
    base = {k: np.concatenate([d[k] for d in ds]) for k in ['labels', 'subject', 'quality']}
    base['allocs'] = ds[0]['allocs']
    logp = L - np.log(np.exp(L - L.max(-1, keepdims=True)).sum(-1, keepdims=True)) - L.max(-1, keepdims=True)
    probs = np.exp(logp).mean(0)                               # average over repeats: N x 124 x 27
    allocs = [tuple(int(v) for v in a) for a in base['allocs']]
    N = probs.shape[0]
    P = np.full((N, len(STATES), 27), 1 / 27, dtype=np.float32)  # state (0,0,0): uniform guess
    for j, a in enumerate(allocs):
        P[:, SID[a]] = probs[:, j]
    return P, base['labels'], base['subject'], base['quality']


def upgrade_table(tiers):
    """for each state and modality: index of the state one tier up (-1 if not allowed) and its token cost."""
    up = np.full((len(STATES), 3), -1)
    dc = np.zeros((len(STATES), 3))
    for s in STATES:
        if any(k not in tiers for k in s):
            continue
        for m in range(3):
            k = s[m]
            nk = {t: tiers[i + 1] for i, t in enumerate(tiers[:-1])}.get(k)
            if nk is not None:
                t = list(s)
                t[m] = nk
                up[SID[s], m] = SID[tuple(t)]
                dc[SID[s], m] = nk - k
    return up, dc


FEAT_LOGP = False   # use log class probabilities instead of probabilities as router input
FEAT_COMPACT = True    # class-agnostic top-3 probs generalise across subjects better (router_reg.py)


def features(P, quality, state_ids):
    """router input for (sample, state) pairs: tier one-hots, quality cues of active sensors,
    current class probabilities, max prob, entropy.  P: n x 27 rows for those pairs."""
    st = np.array(STATES)[state_ids]                           # n x 3
    onehot = np.concatenate([(st[:, [m]] == np.array(TIERS)[None]).astype(np.float32) for m in range(3)], 1)
    q = np.stack([np.log1p(quality[:, 2] * 100), np.log1p(quality[:, 1]), np.log1p(quality[:, 0])], 1)
    q = q * (st > 0)                                           # only sensors that are on are observed
    ent = -(P * np.log(P + 1e-9)).sum(1, keepdims=True)
    cls = np.log(P + 1e-6) if FEAT_LOGP else P
    if FEAT_COMPACT:   # class-agnostic: only the top-3 sorted probabilities
        cls = -np.sort(-P, 1)[:, :3]
    return np.concatenate([onehot, q, cls, P.max(1, keepdims=True), ent], 1).astype(np.float32)


def train_router(P, quality, gain, up, idx, epochs=6, seed=0, device='cuda', loss='mse', clip=None,
                 cheap_weight=False, hidden=256, dropout=0.0, wd=0.0):
    """gain: N x S x 3 target (MICI of upgrading each modality), masked where up == -1.
    loss 'huber' and clip limit the pull of rare huge -ln p drops; cheap_weight up-weights
    low-token states, which are the ones the greedy router actually visits."""
    torch.manual_seed(seed)
    states = np.where((up >= 0).any(1))[0]
    ii, ss = np.meshgrid(idx, states, indexing='ij')
    ii, ss = ii.ravel(), ss.ravel()
    X = torch.tensor(features(P[ii, ss], quality[ii], ss), device=device)
    g = np.nan_to_num(gain[ii, ss])
    if clip is not None:
        g = np.clip(g, -clip, clip)
    Y = torch.tensor(g, device=device, dtype=torch.float32)
    M = torch.tensor(up[ss] >= 0, device=device, dtype=torch.float32)
    if cheap_weight:
        w = 1.0 / (1.0 + np.array(STATES)[ss].sum(1) / 8.0)
        M = M * torch.tensor(w[:, None] / w.mean(), device=device, dtype=torch.float32)
    lossf = (lambda a, b: torch.nn.functional.huber_loss(a, b, reduction='none', delta=1.0)) if loss == 'huber' \
        else (lambda a, b: (a - b) ** 2)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(hidden, 3)).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for k in range(0, n, 4096):
            b = perm[k:k + 4096]
            loss = (lossf(net((X[b] - mu) / sd), Y[b]) * M[b]).sum() / M[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return lambda x: net((torch.tensor(x, device=device) - mu) / sd).detach().cpu().numpy()


def greedy(P, quality, idx, up, dc, score_fn, lam, true_gain=None):
    """upgrade the modality with the largest predicted gain per token until it falls below lam."""
    n = len(idx)
    state = np.full(n, SID[(0, 0, 0)])
    active = np.ones(n, bool)
    for _ in range(12):
        if not active.any():
            break
        a = np.where(active)[0]
        s = state[a]
        if true_gain is not None:
            g = true_gain[idx[a], s]
        else:
            g = score_fn(features(P[idx[a], s], quality[idx[a]], s))
        ratio = np.where(up[s] >= 0, g / np.maximum(dc[s], 1), -np.inf)
        best = ratio.argmax(1)
        go = ratio[np.arange(len(a)), best] >= lam
        # never stop with nothing switched on
        go |= s == SID[(0, 0, 0)]
        state[a[go]] = up[s[go], best[go]]
        active[a[~go]] = False
    return state


def evaluate(P, y, idx, state):
    tok = np.array(STATES)[state].sum(1)
    acc = (P[idx, state].argmax(1) == y[idx]).mean()
    return float(tok.mean()), float(acc)


def envelope_at(points, x):
    """best accuracy among points with avg tokens <= x."""
    ok = [a for t, a in points if t <= x + 1e-9]
    return max(ok) if ok else float('nan')


def run_split(P, y, quality, subjects, seed, out_dir, gains):
    rng = np.random.RandomState(seed)
    subs = np.array(sorted(set(subjects)))
    rng.shuffle(subs)
    tr_s = set(subs[:len(subs) // 2])
    tr = np.where([s in tr_s for s in subjects])[0]
    te = np.where([s not in tr_s for s in subjects])[0]
    correct = P.argmax(2) == y[:, None]                        # N x S
    tok = np.array(STATES).sum(1)
    real = [i for i, s in enumerate(STATES) if sum(s) > 0]
    curves = {}

    # fixed allocations
    acc_tr, acc_te = correct[tr].mean(0), correct[te].mean(0)
    curves['全满（X-Fi 默认）'] = [(FULL, float(acc_te[SID[(32, 32, 32)]]))]
    curves['平均分配'] = [(3 * k, float(acc_te[SID[(k, k, k)]])) for k in TIERS[1:]]
    pts = []
    for B in sorted(set(tok[real])):
        cand = [i for i in real if tok[i] <= B]
        j = max(cand, key=lambda i: acc_tr[i])                 # chosen on the train half
        pts.append((float(tok[j]), float(acc_te[j])))
    curves['最好的固定分配'] = pts
    onoff = [i for i in real if all(k in (0, 32) for k in STATES[i])]
    pts = []
    for B in sorted(set(tok[onoff])):
        j = max([i for i in onoff if tok[i] <= B], key=lambda i: acc_tr[i])
        pts.append((float(tok[j]), float(acc_te[j])))
    curves['只选模态（固定）'] = pts
    curves['随机分配'] = [(float(T), float(np.mean([acc_te[i] for i in real if tok[i] == T])))
                       for T in sorted(set(tok[real]))]

    # confidence cascade along the best-fixed path: stop once max prob >= tau
    path = []
    for B in sorted(set(tok[real])):
        j = max([i for i in real if tok[i] <= B], key=lambda i: acc_tr[i])
        if not path or acc_tr[j] > acc_tr[path[-1]]:
            path.append(j)
    pts = []
    for tau in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01]:
        st = np.full(len(te), path[-1])
        done = np.zeros(len(te), bool)
        for j in path:
            hit = ~done & (P[te, j].max(1) >= tau)
            st[hit] = j
            done |= hit
        pts.append(evaluate(P, y, te, st))
    curves['置信度级联'] = pts

    # oracle: cheapest correct allocation within a per-sample cap
    pts = []
    for B in sorted(set(tok[real])):
        cand = np.array([i for i in real if tok[i] <= B])
        c = correct[te][:, cand]
        cost = np.where(c, tok[cand][None], 10 ** 6)
        pick = np.where(c.any(1), cand[cost.argmin(1)], cand[np.argmax(acc_tr[cand])])
        pts.append(evaluate(P, y, te, pick))
    curves['oracle（上限）'] = pts

    lams = np.concatenate([[-1e9, 0], np.logspace(-4, 0.3, 30), [1e9]])
    up, dc = upgrade_table(TIERS)
    up2, dc2 = upgrade_table([0, 32])
    routers = {}
    for name, g in gains.items():
        routers[name] = train_router(P, quality, g, up, tr, seed=seed, loss='huber', clip=None if 'PPL' in name else 3.0)
    routers['只选模态（router）'] = train_router(P, quality, gains['MICI router（−ln p）'], up2, tr, seed=seed, loss='huber', clip=3.0)
    for name, f in routers.items():
        u, d = (up2, dc2) if name.startswith('只选模态') else (up, dc)
        scale = 1.0
        if 'PPL' in name:   # PPL drops live on a different scale; sweep thresholds in that scale
            scale = float(np.nanmedian(np.abs(gains[name][tr])) / np.nanmedian(np.abs(gains['MICI router（−ln p）'][tr])))
        curves[name] = [evaluate(P, y, te, greedy(P, quality, te, u, d, f, lam * scale)) for lam in lams]
    curves['真实 MICI 贪心（参考）'] = [evaluate(P, y, te, greedy(P, quality, te, up, dc, None, lam,
                                                          true_gain=gains['MICI router（−ln p）'])) for lam in lams]
    return curves, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out_dir', default=os.path.join(pilot_paths.RESULTS, 'alloc_analysis'))
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    P, y, subjects, quality = load(args.inputs)
    N = len(y)
    nll = -np.log(P[np.arange(N), :, y] + 1e-12)                # N x S
    up, dc = upgrade_table(TIERS)
    safe_up = np.where(up >= 0, up, 0)
    g_nll = np.where(up[None] >= 0, nll[:, :, None] - nll[:, safe_up], np.nan)      # MICI, −ln p version
    inv = 1 / np.clip(P[np.arange(N), :, y], 1e-6, None)
    g_ppl = np.where(up[None] >= 0, inv[:, :, None] - inv[:, safe_up], np.nan)       # PPL (1/p) version
    clip = np.nanpercentile(np.abs(g_ppl), 99.5)
    g_ppl = np.clip(g_ppl, -clip, clip)
    gains = {'MICI router（−ln p）': g_nll, 'MICI router（PPL）': g_ppl}

    summary = {'n_samples': int(N), 'n_subjects': int(len(set(subjects)))}
    correct = P.argmax(2) == y[:, None]
    summary['no_training_pooling_acc'] = {f'{k}/{k}/{k}': float(correct[:, SID[(k, k, k)]].mean()) for k in TIERS[1:]}
    summary['acc_by_allocation'] = {f'D{s[0]}-M{s[1]}-L{s[2]}': float(correct[:, SID[s]].mean())
                                    for s in STATES if sum(s) > 0}
    # do the −ln p and PPL versions pick the same upgrade? (true labels, every sample and state)
    ok = (up >= 0).any(1)
    r1 = np.where(up[None] >= 0, g_nll / np.maximum(dc[None], 1), -np.inf)[:, ok].argmax(2)
    r2 = np.where(up[None] >= 0, g_ppl / np.maximum(dc[None], 1), -np.inf)[:, ok].argmax(2)
    summary['nll_vs_ppl_same_greedy_choice'] = float((r1 == r2).mean())

    all_curves = []
    for seed in range(args.seeds):
        curves, tr, te = run_split(P, y, quality, subjects, seed, args.out_dir, gains)
        all_curves.append(curves)
    names = list(all_curves[0])
    table = {}
    for nm in names:
        row = {}
        for x in (6, 8, 12, 24, 48):
            v = [envelope_at(c[nm], x) for c in all_curves]
            row[f'acc@{x}tok'] = [float(np.mean(v)), float(np.std(v))]
        row['curve_seed0'] = all_curves[0][nm]
        table[nm] = row
    summary['methods'] = table
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for nm in names:
        pts = sorted(all_curves[0][nm])
        xs, ys = [p[0] for p in pts], [p[1] * 100 for p in pts]
        style = dict(marker='o', ms=3)
        if nm.startswith('MICI router（−ln p）'):
            style.update(lw=2.5, color='black')
        if len(pts) == 1:
            ax.scatter(xs, ys, label=nm, zorder=5, s=40)
        else:
            ax.plot(xs, ys, label=nm, **style)
    ax.axvline(24, color='grey', ls=':', lw=0.8)
    ax.axvline(48, color='grey', ls=':', lw=0.8)
    ax.set_xlabel('平均每个样本的 token 数（满额 96）')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('MM-Fi 动作识别：按样本分配 token（测试半，受试者划分 seed 0）')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'tokens_vs_acc.png'), dpi=150)
    print(json.dumps({k: v for k, v in summary.items() if k != 'acc_by_allocation'}, ensure_ascii=False, indent=1)[:6000])


if __name__ == '__main__':
    main()
