"""Fair token accounting for the confidence cascade.

The cascade walks a fixed path of whole allocations (best fixed allocation per budget) and stops
at the first step whose top-class probability >= tau.  Old cost = tokens of the stop allocation.
Fair cost = per sensor, the most tokens it had in any allocation visited up to the stop, summed
(a sensor that was switched on earlier has already been paid for).  The MICI router only ever
moves one sensor up one tier (analyze_alloc.greedy uses upgrade_table), so its final allocation
already equals its per-sensor maximum and its cost is unchanged.
CPU only, uses the saved X-Fi sweep; same splits / tau sweep / envelope rule as analyze_alloc.py."""
import pilot_paths  # noqa: E402
import glob
import json
import os

import numpy as np

import analyze_alloc as A

R = pilot_paths.RESULTS
OUT = os.path.join(R, 'fair_cost')
BUDGETS = [6, 8, 12, 24, 48]
TAUS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01]       # as in analyze_alloc.run_split
LAMS = np.concatenate([[-1e9, 0], np.logspace(-4, 0.3, 30), [1e9]])     # as in analyze_alloc.run_split
GRID = np.arange(4, 61)


def main():
    os.makedirs(OUT, exist_ok=True)
    P, y, subjects, quality = A.load(sorted(glob.glob(os.path.join(R, 'alloc_full_p*.npz'))))
    N = len(y)
    st = np.array(A.STATES)
    tok = st.sum(1)
    real = [i for i, s in enumerate(A.STATES) if sum(s) > 0]
    correct = P.argmax(2) == y[:, None]
    nll = -np.log(P[np.arange(N), :, y] + 1e-12)
    up, dc = A.upgrade_table(A.TIERS)
    # router moves are one-tier upgrades only: every valid upgrade raises exactly one sensor
    valid = up >= 0
    assert all((st[up[s, m]] >= st[s]).all() and (st[up[s, m]] - st[s]).sum() > 0
               for s in range(len(st)) for m in range(3) if valid[s, m])
    safe = np.where(valid, up, 0)
    g = np.where(valid[None], nll[:, :, None] - nll[:, safe], np.nan)

    names = ['MICI router', '置信度级联（原记账）', '置信度级联（公平记账）', '最好的固定分配']
    table = {k: [] for k in names}
    grid_curves = {k: [] for k in names}
    stop_info = {}
    for seed in range(3):
        rng = np.random.RandomState(seed)
        subs = np.array(sorted(set(subjects)))
        rng.shuffle(subs)
        trs = set(subs[:len(subs) // 2])
        tr = np.where([s in trs for s in subjects])[0]
        te = np.where([s not in trs for s in subjects])[0]
        acc_tr, acc_te = correct[tr].mean(0), correct[te].mean(0)

        curves = {}
        fixed = []
        path = []
        for B in sorted(set(tok[real])):
            j = max([i for i in real if tok[i] <= B], key=lambda i: acc_tr[i])
            fixed.append((float(tok[j]), float(acc_te[j])))
            if not path or acc_tr[j] > acc_tr[path[-1]]:
                path.append(j)
        curves['最好的固定分配'] = fixed

        path = np.array(path)
        pmax = np.maximum.accumulate(st[path], axis=0).sum(1)       # fair cost if stopping at step k
        old, fair = [], []
        stops = {}
        for tau in TAUS:
            stop = np.full(len(te), len(path) - 1)
            done = np.zeros(len(te), bool)
            for k, j in enumerate(path):
                hit = ~done & (P[te, j].max(1) >= tau)
                stop[hit] = k
                done |= hit
            a = float(correct[te, path[stop]].mean())
            old.append((float(tok[path[stop]].mean()), a))
            fair.append((float(pmax[stop].mean()), a))
            stops[tau] = np.bincount(stop, minlength=len(path)) / len(te)
        curves['置信度级联（原记账）'] = old
        curves['置信度级联（公平记账）'] = fair

        f = A.train_router(P, quality, g, up, tr, seed=seed, device='cpu', loss='huber', clip=3.0)
        curves['MICI router'] = [A.evaluate(P, y, te, A.greedy(P, quality, te, up, dc, f, l)) for l in LAMS]

        for k in names:
            table[k].append([A.envelope_at(curves[k], b) for b in BUDGETS])
            grid_curves[k].append([A.envelope_at(curves[k], b) for b in GRID])
        stop_info[seed] = {
            'path': ['D%d-M%d-L%d' % A.STATES[j] for j in path],
            'old_cost_per_step': [int(tok[j]) for j in path],
            'fair_cost_per_step': [int(c) for c in pmax],
            'stop_share_by_tau': {str(t): [round(float(v) * 100, 1) for v in s] for t, s in stops.items()},
            'cascade_points': {'old': old, 'fair': fair},
        }
        print(f'seed {seed} done', flush=True)

    summary = {k: {f'@{b}': [round(float(np.mean(np.array(v)[:, i])) * 100, 1), round(float(np.std(np.array(v)[:, i])) * 100, 1)]
                   for i, b in enumerate(BUDGETS)} for k, v in table.items()}
    json.dump({'table': summary, 'cascade_detail': stop_info}, open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for k, v in summary.items():
        print(f'{k:16s}', ' '.join(f'{b}:{m:.1f}±{s:.1f}' for b, (m, s) in v.items()))
    for seed, d in stop_info.items():
        print(seed, d['path'], d['old_cost_per_step'], d['fair_cost_per_step'])
        for t in ['0.8', '0.9', '0.95', '0.99']:
            print('  tau', t, d['stop_share_by_tau'][t])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(7.5, 5))
    style = {'MICI router': dict(color='black', lw=2.5), '置信度级联（原记账）': dict(color='C4', ls='--'),
             '置信度级联（公平记账）': dict(color='C4'), '最好的固定分配': dict(color='C1')}
    for k, s in style.items():
        m = np.nanmean(np.array(grid_curves[k], dtype=float), 0) * 100
        ax.plot(GRID, m, label=k, **s)
    ax.set_xlabel('预算：平均每个样本最多用的 token 数（满额 96）')
    ax.set_ylabel('准确率 (%)')
    ax.set_title('公平记账前后（3 种按人划分的平均）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fair_cost_curve.png'), dpi=150)


if __name__ == '__main__':
    main()
