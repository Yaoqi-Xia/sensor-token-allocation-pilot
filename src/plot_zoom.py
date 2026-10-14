"""Zoomed token-vs-accuracy plot (0-20 tokens, where the methods differ) from summary.json."""
import os
import pilot_paths  # noqa: E402
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(pilot_paths.RESULTS, 'alloc_analysis')
s = json.load(open(f'{d}/summary.json', encoding='utf-8'))
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
show = {'MICI router（−ln p）': dict(color='black', lw=2.5),
        '置信度级联': dict(color='C4'),
        '最好的固定分配': dict(color='C1'),
        'oracle（上限）': dict(color='C5', ls='--'),
        '真实 MICI 贪心（参考）': dict(color='C8', ls='--')}
fig, ax = plt.subplots(figsize=(7, 4.8))
for nm, st in show.items():
    pts = sorted(p for p in s['methods'][nm]['curve_seed0'] if p[0] <= 20)
    ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], marker='o', ms=3, label=nm, **st)
full = s['methods']['全满（X-Fi 默认）']['curve_seed0'][0]
ax.axhline(full[1] * 100, color='C0', ls=':', label=f'全满 96 token（{full[1] * 100:.1f}%）')
ax.set_xlim(3, 20)
ax.set_xlabel('平均每个样本的 token 数（满额 96）')
ax.set_ylabel('准确率 (%)')
ax.set_title('低预算区放大（测试半，受试者划分 seed 0）')
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(f'{d}/tokens_vs_acc_zoom.png', dpi=150)
print('saved')
