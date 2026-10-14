"""Which hindsight signal is the best basis for allocating tokens?  Same greedy rule, three
definitions of an upgrade's value:  true MICI (drop in -ln p of the true class), change in
correctness (0/1), and change in top-class confidence (label-free).  Uses the saved sweep only."""
import os
import pilot_paths  # noqa: E402
import glob
import json

import numpy as np

import analyze_alloc as A

BUDGETS = [5, 6, 8, 12, 16, 24]
LAMS = np.concatenate([[-1e9, 0], np.logspace(-5, 0.5, 40), [1e9]])

P, y, subjects, quality = A.load(sorted(glob.glob(os.path.join(pilot_paths.RESULTS, 'alloc_full_p*.npz'))))
N = len(y)
up, dc = A.upgrade_table(A.TIERS)
safe = np.where(up >= 0, up, 0)
valid = up[None] >= 0

nll = -np.log(P[np.arange(N), :, y] + 1e-12)                  # N x S
corr = (P.argmax(2) == y[:, None]).astype(np.float32)          # N x S
conf = P.max(2)                                                 # N x S

signals = {
    '1 真实 MICI（−ln p 下降）': nll[:, :, None] - nll[:, safe],
    '2 答对与否的变化（0/1）': corr[:, safe] - corr[:, :, None],
    '3 置信度提升（不看答案）': conf[:, safe] - conf[:, :, None],
}
idx = np.arange(N)
out = {}
for name, g in signals.items():
    g = np.where(valid, g, np.nan)
    g = g - 1e-6 * dc[None]          # ties (common for the 0/1 signal) go to the cheaper upgrade
    curve = [A.evaluate(P, y, idx, A.greedy(P, quality, idx, up, dc, None, lam, true_gain=g)) for lam in LAMS]
    out[name] = {f'@{b}': round(A.envelope_at(curve, b) * 100, 1) for b in BUDGETS}
    out[name]['curve'] = curve
    print(name, {k: v for k, v in out[name].items() if k != 'curve'}, flush=True)

with open(os.path.join(pilot_paths.RESULTS, 'signal_compare.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
