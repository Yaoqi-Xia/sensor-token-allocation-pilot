"""Router round 2: add each active sensor's encoder summary (PCA-32) to the router input, choose
hyper-parameters on held-out subjects inside the train half, evaluate once on the test half.
Also the attention-allocation baseline (tokens split by X-Fi's fusion attention share)."""
import pilot_paths  # noqa: E402
import glob
import json
import os

import numpy as np

import analyze_alloc as A

R = pilot_paths.RESULTS
OUT = os.path.join(R, 'router_v2')
BUDGETS = [5, 6, 8, 12, 16, 24, 48]
SEL_BUDGETS = [6, 8, 12]
LAMS = np.concatenate([[-1e9, 0], np.logspace(-4, 0.3, 30), [1e9]])
CONFIGS = {
    'h256': dict(loss='huber', clip=3.0, epochs=4),
    'h256-6ep': dict(loss='huber', clip=3.0, epochs=6),
    'h128-drop0.3-wd': dict(loss='huber', clip=3.0, hidden=128, dropout=0.3, wd=1e-2, epochs=4),
    'h64-drop0.3-wd-2ep': dict(loss='huber', clip=3.0, hidden=64, dropout=0.3, wd=1e-2, epochs=2),
}

_base_features = A.features
Z = None          # N x 3 x d PCA summaries, set per split
USE_Z = False


def features_ext(P, qx, state_ids):
    """qx = [quality (3) | sample index (1)]: base router input plus PCA summaries of sensors that are on."""
    base = _base_features(P, qx[:, :3], state_ids)
    if not USE_Z:
        return base
    st = np.array(A.STATES)[state_ids]
    z = Z[qx[:, 3].astype(int)] * (st > 0)[:, :, None]
    return np.concatenate([base, z.reshape(len(z), -1)], 1).astype(np.float32)


A.features = features_ext


def pca(X, d=32, fit_rows=None):
    Xf = X[fit_rows].astype(np.float32)
    mu = Xf.mean(0)
    sub = Xf[np.random.RandomState(0).choice(len(Xf), min(20000, len(Xf)), replace=False)] - mu
    _, s, vt = np.linalg.svd(sub, full_matrices=False)
    W = vt[:d].T / (s[:d] / np.sqrt(len(sub)))          # whitened components
    return (X.astype(np.float32) - mu) @ W


def curve(P, y, qx, idx, up, dc, f):
    return [A.evaluate(P, y, idx, A.greedy(P, qx, idx, up, dc, f, l)) for l in LAMS]


def attention_baseline(P, y, att, idx):
    allocs = np.array([s for s in A.STATES if sum(s) > 0])
    ids = np.array([A.SID[tuple(s)] for s in allocs])
    tok = allocs.sum(1)
    pts = []
    for B in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
        ok = tok <= B
        target = att[idx][:, None, :] * B                                   # n x 1 x 3
        dist = np.abs(allocs[ok][None] - target).sum(2)                     # n x m
        dist += 1e-3 * (B - tok[ok])[None]                                  # prefer using the budget
        pick = ids[ok][dist.argmin(1)]
        pts.append(A.evaluate(P, y, idx, pick))
    return pts


def main():
    global Z, USE_Z
    os.makedirs(OUT, exist_ok=True)
    P, y, subjects, quality = A.load(sorted(glob.glob(os.path.join(R, 'alloc_full_p*.npz'))))
    F = [np.load(p) for p in sorted(glob.glob(os.path.join(R, 'feats_full_p*.npz')))]
    feats = np.concatenate([f['feats'] for f in F]).astype(np.float32)        # N x 3 x 512
    att = np.concatenate([f['att'] for f in F])
    assert (np.concatenate([f['subject'] for f in F]) == subjects).all()
    N = len(y)
    qx = np.concatenate([quality, np.arange(N)[:, None]], 1).astype(np.float32)
    nll = -np.log(P[np.arange(N), :, y] + 1e-12)
    up, dc = A.upgrade_table(A.TIERS)
    safe = np.where(up >= 0, up, 0)
    g = np.where(up[None] >= 0, nll[:, :, None] - nll[:, safe], np.nan)
    correct = P.argmax(2) == y[:, None]
    tok = np.array(A.STATES).sum(1)
    real = [i for i, s in enumerate(A.STATES) if sum(s) > 0]

    results, chosen = {}, {}
    for seed in range(3):
        rng = np.random.RandomState(seed)
        subs = np.array(sorted(set(subjects)))
        rng.shuffle(subs)
        te_s, tr_s = set(subs[:len(subs) // 2]), subs[len(subs) // 2:]
        va_s = set(tr_s[:max(2, len(tr_s) // 5)])
        te = np.where([s in te_s for s in subjects])[0]
        tr = np.where([s not in te_s for s in subjects])[0]
        va = np.where([s in va_s for s in subjects])[0]
        fit = np.where([(s not in te_s) and (s not in va_s) for s in subjects])[0]

        curves = {}
        acc_tr = correct[tr].mean(0)
        pts = []
        for B in sorted(set(tok[real])):
            j = max([i for i in real if tok[i] <= B], key=lambda i: acc_tr[i])
            pts.append((float(tok[j]), float(correct[te, j].mean())))
        curves['最好的固定分配'] = pts
        path = []
        for B in sorted(set(tok[real])):
            j = max([i for i in real if tok[i] <= B], key=lambda i: acc_tr[i])
            if not path or acc_tr[j] > acc_tr[path[-1]]:
                path.append(j)
        pts = []
        for tau in np.concatenate([np.linspace(0.2, 0.95, 16), [0.97, 0.98, 0.99, 0.995, 0.999, 1.01]]):
            st = np.full(len(te), path[-1])
            done = np.zeros(len(te), bool)
            for j in path:
                hit = ~done & (P[te, j].max(1) >= tau)
                st[hit] = j
                done |= hit
            pts.append(A.evaluate(P, y, te, st))
        curves['置信度级联'] = pts
        curves['按 attention 分配（不计探测成本）'] = attention_baseline(P, y, att, te)

        for variant, use_z in [('router（原输入）', False), ('router + 编码器特征', True)]:
            USE_Z = use_z
            # PCA fitted on the fitting subjects only, then on the whole train half for the final model
            Z = np.stack([pca(feats[:, m], fit_rows=fit) for m in range(3)], 1)
            best, best_score = None, -1
            for cname, kw in CONFIGS.items():
                f = A.train_router(P, qx, g, up, fit, seed=seed, **kw)
                c = curve(P, y, qx, va, up, dc, f)
                score = np.mean([A.envelope_at(c, b) for b in SEL_BUDGETS])
                print(f'seed {seed} {variant} {cname}: val {score * 100:.2f}', flush=True)
                if score > best_score:
                    best, best_score = cname, score
            chosen[f'{seed}/{variant}'] = best
            Z = np.stack([pca(feats[:, m], fit_rows=tr) for m in range(3)], 1)
            f = A.train_router(P, qx, g, up, tr, seed=seed, **CONFIGS[best])
            curves[variant] = curve(P, y, qx, te, up, dc, f)
        curves['真实 MICI 贪心（参考）'] = [A.evaluate(P, y, te, A.greedy(P, qx, te, up, dc, None, l, true_gain=g)) for l in LAMS]
        for k, c in curves.items():
            results.setdefault(k, {'acc': [], 'curve_seed0': c if seed == 0 else None})
            results[k]['acc'].append([A.envelope_at(c, b) for b in BUDGETS])
            if seed == 0:
                results[k]['curve_seed0'] = c

    table = {k: {f'@{b}': [round(float(np.mean(np.array(v['acc'])[:, i])) * 100, 1),
                           round(float(np.std(np.array(v['acc'])[:, i])) * 100, 1)] for i, b in enumerate(BUDGETS)}
             for k, v in results.items()}
    json.dump({'table': table, 'chosen_configs': chosen, 'mean_attention_share_DML': att.mean(0).tolist(),
               'curves_seed0': {k: v['curve_seed0'] for k, v in results.items()}},
              open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    for k, v in table.items():
        print(f'{k:28s}', ' '.join(f'{b}:{m:.1f}±{s:.1f}' for b, (m, s) in v.items()))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7.5, 5))
    style = {'router + 编码器特征': dict(color='black', lw=2.5), 'router（原输入）': dict(color='grey'),
             '置信度级联': dict(color='C4'), '最好的固定分配': dict(color='C1'),
             '按 attention 分配（不计探测成本）': dict(color='C2'), '真实 MICI 贪心（参考）': dict(color='C8', ls='--')}
    for k, st in style.items():
        pts = sorted(p for p in results[k]['curve_seed0'] if p[0] <= 50)
        ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], marker='o', ms=3, label=k, **st)
    ax.set_xlabel('平均每个样本的 token 数（满额 96）')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('router 第二轮（测试半，受试者划分 seed 0）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'tokens_vs_acc_v2.png'), dpi=150)


if __name__ == '__main__':
    main()
