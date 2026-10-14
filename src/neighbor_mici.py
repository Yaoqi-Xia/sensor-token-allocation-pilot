"""Neighbour-averaged MICI (idea borrowed from wICI's averaging over probe samples).

Variant A (kNN router): at state s, predicted MICI(s, a) = kernel-weighted mean of the true MICI of
the K nearest training samples in the same state s, compared on the 8 non-tier router inputs
(3 quality cues + 5 prediction features), standardised per state on the training rows.
Variant B: replace every training label by the same neighbour average, then train the v1 router.
K in {10, 50, 200} x {gaussian kernel, uniform} chosen on inner validation persons; test half once.
Same splits as analyze_alloc / fair_cost (train = first half of shuffled persons).  CPU only."""
import pilot_paths  # noqa: E402
import glob
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import analyze_alloc as A

R = pilot_paths.RESULTS
OUT = os.path.join(R, 'neighbor_mici')
BUDGETS = [6, 8, 12, 24, 48]
TAUS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01]
LAMS = np.concatenate([[-1e9, 0], np.logspace(-4, 0.3, 30), [1e9]])
SEL_BUDGETS = [6, 8, 12]
CONFIGS = [(k, w) for k in (10, 50, 200) for w in ('kernel', 'uniform')]
GRID = np.arange(4, 61)
NS = len(A.STATES)
TIERS = np.array(A.TIERS)


def state_of(X):
    """decode state ids from the 15 tier one-hot columns of the router input."""
    t = X[:, :15].reshape(-1, 3, 5).argmax(2)
    return t[:, 0] * 25 + t[:, 1] * 5 + t[:, 2]          # STATES = product(TIERS, repeat=3)


class KNN:
    """per-state KD-trees over the 8 non-tier inputs of the given (training) samples."""
    def __init__(self, P, quality, g, idx):
        self.trees, self.labels, self.mu, self.sd = {}, {}, {}, {}
        for s in range(NS):
            X = A.features(P[idx, s], quality[idx], np.full(len(idx), s))[:, 15:]
            mu, sd = X.mean(0), X.std(0) + 1e-6
            self.trees[s] = cKDTree((X - mu) / sd)
            self.labels[s] = np.nan_to_num(g[idx, s])
            self.mu[s], self.sd[s] = mu, sd

    def predict(self, X, states, k, weight, exclude_self=False):
        out = np.zeros((len(X), 3), np.float32)
        for s in np.unique(states):
            r = np.where(states == s)[0]
            q = (X[r, 15:] - self.mu[s]) / self.sd[s]
            d, j = self.trees[s].query(q, k=k)
            if k == 1:
                d, j = d[:, None], j[:, None]
            if weight == 'kernel':
                h = np.median(d, 1, keepdims=True) + 1e-6
                w = np.exp(-d ** 2 / (2 * h ** 2))
            else:
                w = np.ones_like(d)
            w = w / w.sum(1, keepdims=True)
            out[r] = (w[:, :, None] * self.labels[s][j]).sum(1)
        return out


def greedy_log(P, quality, idx, up, dc, f, lam, g):
    """A.greedy plus a log of (predicted, true) MICI for every visited (state, upgrade)."""
    n = len(idx)
    state = np.full(n, A.SID[(0, 0, 0)])
    active = np.ones(n, bool)
    pred_all, true_all, top_hit = [], [], []
    for _ in range(12):
        if not active.any():
            break
        a = np.where(active)[0]
        s = state[a]
        gp = f(A.features(P[idx[a], s], quality[idx[a]], s))
        valid = up[s] >= 0
        gt = g[idx[a], s]
        pred_all.append(gp[valid])
        true_all.append(gt[valid])
        rp = np.where(valid, gp / np.maximum(dc[s], 1), -np.inf)
        rt = np.where(valid, np.nan_to_num(gt) / np.maximum(dc[s], 1), -np.inf)
        top_hit.append(rp.argmax(1) == rt.argmax(1))
        best = rp.argmax(1)
        go = (rp[np.arange(len(a)), best] >= lam) | (s == A.SID[(0, 0, 0)])
        state[a[go]] = up[s[go], best[go]]
        active[a[~go]] = False
    t, acc = A.evaluate(P, y_global, idx, state)
    return (t, acc), np.concatenate(pred_all), np.concatenate(true_all), np.concatenate(top_hit)


def curve(P, quality, idx, up, dc, f, g, log_at=8):
    pts, logs = [], []
    for lam in LAMS:
        p, pr, tr_, hit = greedy_log(P, quality, idx, up, dc, f, lam, g)
        pts.append(p)
        logs.append((p[0], pr, tr_, hit))
    # diagnostics at the operating point whose mean tokens is closest to log_at
    t0, pr, tr_, hit = min(logs, key=lambda z: abs(z[0] - log_at))
    rho = spearmanr(pr, tr_).correlation
    return pts, {'tokens': float(t0), 'spearman': float(rho), 'top1_agree': float(hit.mean())}


def main():
    global y_global
    os.makedirs(OUT, exist_ok=True)
    A.FEAT_COMPACT = True
    P, y, subjects, quality = A.load(sorted(glob.glob(os.path.join(R, 'alloc_full_p*.npz'))))
    y_global = y
    N = len(y)
    st = np.array(A.STATES)
    tok = st.sum(1)
    real = [i for i, s in enumerate(A.STATES) if sum(s) > 0]
    correct = P.argmax(2) == y[:, None]
    nll = -np.log(P[np.arange(N), :, y] + 1e-12)
    up, dc = A.upgrade_table(A.TIERS)
    safe = np.where(up >= 0, up, 0)
    g = np.where(up[None] >= 0, nll[:, :, None] - nll[:, safe], np.nan)

    names = ['v1 router', '版本 A：kNN router', '版本 B：平滑标签 + v1 router', '置信度级联（公平记账）', '最好的固定分配']
    table = {k: [] for k in names}
    grid = {k: [] for k in names}
    diag = {k: [] for k in names[:3]}
    chosen = {}
    for seed in range(3):
        t_start = time.time()
        rng = np.random.RandomState(seed)
        subs = np.array(sorted(set(subjects)))
        rng.shuffle(subs)
        tr_s = subs[:len(subs) // 2]
        va_s = set(tr_s[:max(2, len(tr_s) // 5)])
        tr_set = set(tr_s)
        tr = np.where([s in tr_set for s in subjects])[0]
        te = np.where([s not in tr_set for s in subjects])[0]
        va = np.where([s in va_s for s in subjects])[0]
        fit = np.where([(s in tr_set) and (s not in va_s) for s in subjects])[0]
        acc_tr, acc_te = correct[tr].mean(0), correct[te].mean(0)
        curves = {}

        # baselines (identical to fair_cost.py)
        fixed, path = [], []
        for B in sorted(set(tok[real])):
            j = max([i for i in real if tok[i] <= B], key=lambda i: acc_tr[i])
            fixed.append((float(tok[j]), float(acc_te[j])))
            if not path or acc_tr[j] > acc_tr[path[-1]]:
                path.append(j)
        curves['最好的固定分配'] = fixed
        path = np.array(path)
        pmax = np.maximum.accumulate(st[path], axis=0).sum(1)
        casc = []
        for tau in TAUS:
            stop = np.full(len(te), len(path) - 1)
            done = np.zeros(len(te), bool)
            for k, j in enumerate(path):
                hit = ~done & (P[te, j].max(1) >= tau)
                stop[hit] = k
                done |= hit
            casc.append((float(pmax[stop].mean()), float(correct[te, path[stop]].mean())))
        curves['置信度级联（公平记账）'] = casc

        # v1 router (same as fair_cost.py)
        f = A.train_router(P, quality, g, up, tr, seed=seed, device='cpu', loss='huber', clip=3.0)
        curves['v1 router'], diag_v1 = curve(P, quality, te, up, dc, f, g)
        diag['v1 router'].append(diag_v1)

        # Variant A: choose K / weighting on validation persons with neighbours from the fit persons
        knn_fit = KNN(P, quality, g, fit)
        best, best_sc = None, -1
        for k, w in CONFIGS:
            fA = (lambda X, k=k, w=w: knn_fit.predict(X, state_of(X), k, w))
            c = [A.evaluate(P, y, va, A.greedy(P, quality, va, up, dc, fA, l)) for l in LAMS]
            sc = np.mean([A.envelope_at(c, b) for b in SEL_BUDGETS])
            print(f'seed {seed} A K={k} {w}: val {sc * 100:.2f}', flush=True)
            if sc > best_sc:
                best, best_sc = (k, w), sc
        chosen[f'{seed}/A'] = best
        knn_tr = KNN(P, quality, g, tr)
        fA = (lambda X, k=best[0], w=best[1]: knn_tr.predict(X, state_of(X), k, w))
        curves['版本 A：kNN router'], dA = curve(P, quality, te, up, dc, fA, g)
        diag['版本 A：kNN router'].append(dA)

        # Variant B: smoothed labels; selection uses a shorter (2-epoch) training to save CPU time
        def smoothed(idx, knn, k, w):
            gs = g.copy()
            for s in range(NS):
                X = A.features(P[idx, s], quality[idx], np.full(len(idx), s))
                gs[idx, s] = np.where(up[s][None] >= 0, knn.predict(X, np.full(len(idx), s), k, w), np.nan)
            return gs
        best, best_sc = None, -1
        for k, w in CONFIGS:
            gs = smoothed(fit, knn_fit, k, w)
            fB = A.train_router(P, quality, gs, up, fit, seed=seed, device='cpu', loss='huber', clip=3.0, epochs=2)
            c = [A.evaluate(P, y, va, A.greedy(P, quality, va, up, dc, fB, l)) for l in LAMS]
            sc = np.mean([A.envelope_at(c, b) for b in SEL_BUDGETS])
            print(f'seed {seed} B K={k} {w}: val {sc * 100:.2f}', flush=True)
            if sc > best_sc:
                best, best_sc = (k, w), sc
        chosen[f'{seed}/B'] = best
        gs = smoothed(tr, knn_tr, *best)
        fB = A.train_router(P, quality, gs, up, tr, seed=seed, device='cpu', loss='huber', clip=3.0)
        curves['版本 B：平滑标签 + v1 router'], dB = curve(P, quality, te, up, dc, fB, g)
        diag['版本 B：平滑标签 + v1 router'].append(dB)

        for k in names:
            table[k].append([A.envelope_at(curves[k], b) for b in BUDGETS])
            grid[k].append([A.envelope_at(curves[k], b) for b in GRID])
        print(f'seed {seed} done in {(time.time() - t_start) / 60:.1f} min', flush=True)

    summary = {k: {f'@{b}': [round(float(np.mean(np.array(v)[:, i])) * 100, 1), round(float(np.std(np.array(v)[:, i])) * 100, 1)]
                   for i, b in enumerate(BUDGETS)} for k, v in table.items()}
    dsum = {k: {m: [round(float(np.mean([d[m] for d in v])), 3), round(float(np.std([d[m] for d in v])), 3)]
                for m in ('tokens', 'spearman', 'top1_agree')} for k, v in diag.items()}
    json.dump({'table': summary, 'diagnostics': dsum, 'chosen': {k: list(v) for k, v in chosen.items()}},
              open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k, v in summary.items():
        print(f'{k:22s}', ' '.join(f'{b}:{m:.1f}±{s:.1f}' for b, (m, s) in v.items()))
    for k, v in dsum.items():
        print(k, v)
    print('chosen', chosen)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7.5, 5))
    style = {'v1 router': dict(color='grey'), '版本 A：kNN router': dict(color='C0', lw=2),
             '版本 B：平滑标签 + v1 router': dict(color='black', lw=2.5),
             '置信度级联（公平记账）': dict(color='C4'), '最好的固定分配': dict(color='C1')}
    for k, s in style.items():
        m = np.array(grid[k], dtype=float)
        ok = ~np.isnan(m).any(0)
        ax.plot(GRID[ok], m[:, ok].mean(0) * 100, label=k, **s)
    ax.set_xlabel('预算：平均每个样本最多用的 token 数（满额 96）')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('邻居平均 MICI（3 种按人划分的平均）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'neighbor_mici_curve.png'), dpi=150)


if __name__ == '__main__':
    main()
