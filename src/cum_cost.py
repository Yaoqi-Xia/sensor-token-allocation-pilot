"""Cumulative token cost: count every X-Fi pass a method needs for a sample (saved sweep only).

Cost A (as before): tokens of the final allocation (cascade: fair per-sensor-max accounting).
Cost B (new): sum of the tokens of every X-Fi pass.
  router  : one pass after each upgrade (the empty start needs no pass)
  cascade : one pass per rung visited, up to and including the stop rung
  best fixed / even / random / all-full : one pass
  attention-proportional : the 4-4-4 probe pass that produced the attention scores (12 tokens)
                           plus one pass at the chosen allocation
Same 3 person splits / router settings / tau sweep / envelope rule as fair_cost.py."""
import pilot_paths  # noqa: E402
import glob
import json
import os

import numpy as np
import torch

import analyze_alloc as A

R = pilot_paths.RESULTS
OUT = os.path.join(R, 'cum_cost')
BUDGETS = [6, 8, 12, 24, 48]
TAUS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01]
LAMS = np.concatenate([[-1e9, 0], np.logspace(-4, 0.3, 30), [1e9]])
GRID = np.arange(4, 101)
ST = np.array(A.STATES)
TOK = ST.sum(1)


def greedy_path(P, quality, idx, up, dc, f, lam):
    """A.greedy, also returning the cumulative tokens over all passes and the number of passes."""
    n = len(idx)
    state = np.full(n, A.SID[(0, 0, 0)])
    cum = np.zeros(n)
    npass = np.zeros(n, int)
    seq = [[] for _ in range(n)]
    active = np.ones(n, bool)
    for _ in range(12):
        if not active.any():
            break
        a = np.where(active)[0]
        s = state[a]
        gp = f(A.features(P[idx[a], s], quality[idx[a]], s))
        ratio = np.where(up[s] >= 0, gp / np.maximum(dc[s], 1), -np.inf)
        best = ratio.argmax(1)
        go = (ratio[np.arange(len(a)), best] >= lam) | (s == A.SID[(0, 0, 0)])
        ga = a[go]
        state[ga] = up[s[go], best[go]]
        cum[ga] += TOK[state[ga]]
        npass[ga] += 1
        for i in ga:
            seq[i].append(A.STATES[state[i]])
        active[a[~go]] = False
    return state, cum, npass, seq


def main():
    os.makedirs(OUT, exist_ok=True)
    A.FEAT_COMPACT = True
    P, y, subjects, quality = A.load(sorted(glob.glob(os.path.join(R, 'alloc_full_p*.npz'))))
    att = np.concatenate([np.load(p)['att'] for p in sorted(glob.glob(os.path.join(R, 'feats_full_p*.npz')))])
    N = len(y)
    real = [i for i, s in enumerate(A.STATES) if sum(s) > 0]
    correct = P.argmax(2) == y[:, None]
    nll = -np.log(P[np.arange(N), :, y] + 1e-12)
    up, dc = A.upgrade_table(A.TIERS)
    safe = np.where(up >= 0, up, 0)
    g = np.where(up[None] >= 0, nll[:, :, None] - nll[:, safe], np.nan)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    methods = ['MICI router', '置信度级联', '最好的固定分配', '平均分配', '随机分配', '按注意力分配', '全开（96）']
    tabA = {m: [] for m in methods}
    tabB = {m: [] for m in methods}
    gridB = {m: [] for m in methods}
    paths = {}
    full_accs = []
    for seed in range(3):
        rng = np.random.RandomState(seed)
        subs = np.array(sorted(set(subjects)))
        rng.shuffle(subs)
        trs = set(subs[:len(subs) // 2])
        tr = np.where([s in trs for s in subjects])[0]
        te = np.where([s not in trs for s in subjects])[0]
        acc_tr, acc_te = correct[tr].mean(0), correct[te].mean(0)
        cA, cB = {}, {}

        fixed, path = [], []
        for B in sorted(set(TOK[real])):
            j = max([i for i in real if TOK[i] <= B], key=lambda i: acc_tr[i])
            fixed.append((float(TOK[j]), float(acc_te[j])))
            if not path or acc_tr[j] > acc_tr[path[-1]]:
                path.append(j)
        cA['最好的固定分配'] = cB['最好的固定分配'] = fixed
        even = [(3.0 * k, float(acc_te[A.SID[(k, k, k)]])) for k in A.TIERS[1:]]
        cA['平均分配'] = cB['平均分配'] = even
        rnd = [(float(T), float(np.mean([acc_te[i] for i in real if TOK[i] == T]))) for T in sorted(set(TOK[real]))]
        cA['随机分配'] = cB['随机分配'] = rnd
        full = [(96.0, float(acc_te[A.SID[(32, 32, 32)]]))]
        cA['全开（96）'] = cB['全开（96）'] = full
        full_accs.append(full[0][1])

        # attention-proportional: probe 4-4-4 (12 tokens) + final allocation
        allocs = np.array([s for s in A.STATES if sum(s) > 0])
        ids = np.array([A.SID[tuple(s)] for s in allocs])
        tok_a = allocs.sum(1)
        pa, pb = [], []
        for B in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
            ok = tok_a <= B
            dist = np.abs(allocs[ok][None] - att[te][:, None, :] * B).sum(2) + 1e-3 * (B - tok_a[ok])[None]
            pick = ids[ok][dist.argmin(1)]
            acc = float(correct[te, pick].mean())
            pa.append((float(TOK[pick].mean()), acc))
            pb.append((float(TOK[pick].mean() + 12), acc))
        cA['按注意力分配'], cB['按注意力分配'] = pa, pb

        # cascade: A = fair per-sensor max; B = sum over visited rungs
        path = np.array(path)
        fairc = np.maximum.accumulate(ST[path], axis=0).sum(1)
        cumc = np.cumsum(TOK[path])
        ca, cb, cpaths = [], [], {}
        for tau in TAUS:
            stop = np.full(len(te), len(path) - 1)
            done = np.zeros(len(te), bool)
            for k, j in enumerate(path):
                hit = ~done & (P[te, j].max(1) >= tau)
                stop[hit] = k
                done |= hit
            acc = float(correct[te, path[stop]].mean())
            ca.append((float(fairc[stop].mean()), acc))
            cb.append((float(cumc[stop].mean()), acc))
            cpaths[tau] = (float(TOK[path[stop]].mean()), float((stop + 1).mean()), np.bincount(stop, minlength=len(path)) / len(te))
        cA['置信度级联'], cB['置信度级联'] = ca, cb

        # router
        f = A.train_router(P, quality, g, up, tr, seed=seed, device=dev, loss='huber', clip=3.0)
        ra, rb, rpaths = [], [], {}
        for lam in LAMS:
            st, cum, npass, seq = greedy_path(P, quality, te, up, dc, f, lam)
            acc = float(correct[te, st].mean())
            ra.append((float(TOK[st].mean()), acc))
            rb.append((float(cum.mean()), acc))
            rpaths[lam] = (float(TOK[st].mean()), float(npass.mean()), seq)
        cA['MICI router'], cB['MICI router'] = ra, rb

        for m in methods:
            tabA[m].append([A.envelope_at(cA[m], b) for b in BUDGETS])
            tabB[m].append([A.envelope_at(cB[m], b) for b in BUDGETS])
            gridB[m].append([A.envelope_at(cB[m], b) for b in GRID])

        # operating points closest to ~8 final tokens
        lam8 = min(rpaths, key=lambda l: abs(rpaths[l][0] - 8))
        t8, np8, seq8 = rpaths[lam8]
        top = {}
        for sq in seq8:
            key = ' → '.join('%d-%d-%d' % s for s in sq)
            top[key] = top.get(key, 0) + 1
        top = sorted(top.items(), key=lambda z: -z[1])[:6]
        tau8 = min(cpaths, key=lambda t: abs(cpaths[t][0] - 8))
        ct8, cnp8, cdist = cpaths[tau8]
        paths[seed] = {
            'router': {'final_tokens': t8, 'passes_per_sample': np8,
                       'cum_tokens': float(np.mean([sum(sum(s) for s in sq) for sq in seq8])),
                       'top_paths': [(k, round(v / len(seq8) * 100, 1)) for k, v in top]},
            'cascade': {'tau': tau8, 'final_tokens': ct8, 'passes_per_sample': cnp8,
                        'cum_tokens': float((cumc * cdist).sum()),
                        'rungs': ['%d-%d-%d' % A.STATES[j] for j in path],
                        'stop_share': [round(float(v) * 100, 1) for v in cdist]},
        }
        print(f'seed {seed} done', flush=True)
        json.dump({'paths': paths}, open(os.path.join(OUT, 'partial.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def summ(tab):
        return {m: {f'@{b}': [None if np.isnan(np.array(v)[:, i]).any() else round(float(np.mean(np.array(v)[:, i])) * 100, 1),
                              None if np.isnan(np.array(v)[:, i]).any() else round(float(np.std(np.array(v)[:, i])) * 100, 1)]
                    for i, b in enumerate(BUDGETS)} for m, v in tab.items()}
    sA, sB = summ(tabA), summ(tabB)
    json.dump({'costA': sA, 'costB': sB, 'full96_acc': [round(float(np.mean(full_accs)) * 100, 1), round(float(np.std(full_accs)) * 100, 1)], 'paths': paths}, open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for nm, s in [('A 最终分配', sA), ('B 累计', sB)]:
        print('==', nm)
        for m, v in s.items():
            print(f'{m:12s}', ' '.join(f'{b}:{x[0]}' for b, x in v.items()))
    for seed, d in paths.items():
        print(seed, json.dumps(d, ensure_ascii=False)[:900])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {'MICI router': dict(color='black', lw=2.5), '置信度级联': dict(color='C4'), '最好的固定分配': dict(color='C1'),
             '平均分配': dict(color='C0'), '随机分配': dict(color='C3', alpha=0.6), '按注意力分配': dict(color='C2')}
    for m, s in style.items():
        arr = np.array(gridB[m], dtype=float)
        ok = ~np.isnan(arr).any(0)
        ax.plot(GRID[ok], arr[:, ok].mean(0) * 100, label=m, **s)
    ax.scatter([96], [np.mean(full_accs) * 100], color='k', marker='*', s=120, zorder=5, label='全开（96）')
    ax.set_xlabel('平均每个样本累计用的 token 数（每次 X-Fi 运行都算）')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('累计 token 对比（3 种按人划分的平均）')
    ax.set_ylim(40, 90)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'cum_cost_curve.png'), dpi=150)


if __name__ == '__main__':
    main()
