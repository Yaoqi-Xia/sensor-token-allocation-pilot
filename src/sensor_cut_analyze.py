"""Analyse the encoder-cut test against after-projector average pooling (stored sweep), same frames."""
import pilot_paths  # noqa: E402
import glob
import json
import os

import numpy as np

R = pilot_paths.RESULTS
OUT = os.path.join(R, 'sensor_cut')
KS = [4, 8, 16, 32]
FULL = {'depth': 49, 'mmwave': 32, 'lidar': 32}
ALLOC = {'depth': 'D{k}-M0-L0', 'mmwave': 'D0-M{k}-L0', 'lidar': 'D0-M0-L{k}'}
CN = {'depth': '深度', 'mmwave': '毫米波', 'lidar': 'LiDAR'}

d = np.load(os.path.join(OUT, 'sensor_cut_preds.npz'))
preds, labels, names, sub = d['preds'], d['labels'], list(d['names']), d['subset']
# after-projector pooling from the stored sweep (pieces concatenated in sample order)
pieces = [np.load(p) for p in sorted(glob.glob(os.path.join(R, 'alloc_full_p*.npz')))]
anames = list(pieces[0]['alloc_names'])
store_lab = np.concatenate([p['labels'] for p in pieces])
assert (store_lab[sub] == labels).all(), 'frame order mismatch'
pool_pred = {}
for s in FULL:
    for k in KS:
        j = anames.index(ALLOC[s].format(k=k))
        pool_pred[(s, k)] = np.concatenate([p['logits'][0][:, j].astype(np.float32).argmax(1) for p in pieces])[sub]


def oracle(correct_by_k, ks, full_k):
    """correct_by_k: n x len(ks) booleans; smallest k that is correct (full k cost if none)."""
    c = np.array(correct_by_k).T
    anyc = c.any(1)
    first = np.where(anyc, np.array(ks)[c.argmax(1)], full_k)
    acc_fixed = c.mean(0)
    mean_k = first.mean()
    ok = [i for i, k in enumerate(ks) if k <= mean_k]
    best_fixed = (ks[max(ok, key=lambda i: acc_fixed[i])], float(acc_fixed[max(ok, key=lambda i: acc_fixed[i])])) if ok else None
    dist = {str(k): float((anyc & (first == k)).mean()) for k in ks}
    dist['都答错'] = float((~anyc).mean())
    return {'oracle_acc': float(anyc.mean()), 'mean_oracle_k': float(mean_k), 'best_fixed_k_within': best_fixed,
            'full_acc': float(acc_fixed[-1]), 'full_k': int(ks[-1]), 'dist': dist}


res = {'n_frames': int(len(labels)), 'acc': {}, 'oracle': {}}
for s, n in FULL.items():
    ks_cut = KS + ([n] if n not in KS else [])
    cut = [preds[:, names.index(f'{s}|{k}|repeat')] == labels for k in ks_cut]
    zero = {k: float((preds[:, names.index(f'{s}|{k}|zero')] == labels).mean()) for k in ks_cut if k < n}
    pool = [pool_pred[(s, k)] == labels for k in KS]
    res['acc'][s] = {'encoder_cut': {k: float(c.mean()) for k, c in zip(ks_cut, cut)},
                     'encoder_cut_zero_fill': zero,
                     'after_projector_pool': {k: float(c.mean()) for k, c in zip(KS, pool)}}
    res['oracle'][s] = {'encoder_cut': oracle(cut, ks_cut, n), 'after_projector_pool': oracle(pool, KS, 32)}

json.dump(res, open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, s in zip(axes, FULL):
    a = res['acc'][s]
    for key, lab, st in [('encoder_cut', '编码器后截断（重复填充）', dict(color='C3', marker='o', lw=2)),
                         ('after_projector_pool', '投影后平均压缩（原方法）', dict(color='C0', marker='s')),
                         ('encoder_cut_zero_fill', '编码器后截断（补零，参考）', dict(color='grey', marker='x', ls=':'))]:
        ks = sorted(a[key])
        ax.plot(ks, [a[key][k] * 100 for k in ks], label=lab, **st)
    ax.set_title(f'只用{CN[s]}')
    ax.set_xlabel('保留的特征数 k')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(a['encoder_cut']))
    ax.set_xticklabels(sorted(a['encoder_cut']))
    ax.grid(alpha=0.3)
axes[0].set_ylabel('准确率 (%)')
axes[0].legend(fontsize=8)
fig.suptitle(f'传感器小测试：截断位置不同，档位是否起作用（{len(labels)} 帧固定子集）')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sensor_cut_acc_vs_k.png'), dpi=150)
