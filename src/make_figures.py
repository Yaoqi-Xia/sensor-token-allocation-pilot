"""English README figures, drawn only from the small summary files in results/ (no GPU, no raw data)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pilot_paths  # noqa: E402

R = pilot_paths.RESULTS
OUT = os.path.join(os.path.dirname(R), 'figures')


def load(*p):
    with open(os.path.join(R, *p), encoding='utf-8') as f:
        return json.load(f)


def budget_curve():
    t = load('fair_cost', 'summary.json')['table']
    rows = [('MICI router (DIME-style)', 'MICI router', 'k', '-'),
            ('Confidence cascade', '置信度级联（公平记账）', 'tab:purple', '-'),
            ('Best fixed allocation', '最好的固定分配', 'tab:orange', '-')]
    budgets = [6, 8, 12, 24, 48]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for label, key, c, ls in rows:
        m = [t[key][f'@{b}'][0] for b in budgets]
        s = [t[key][f'@{b}'][1] for b in budgets]
        ax.errorbar(budgets, m, yerr=s, color=c, ls=ls, marker='o', capsize=3, label=label)
    ax.axhline(77.9, color='tab:blue', ls=':', label='All sensors, 96 tokens (77.9)')
    ax.set_xscale('log', base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(budgets)
    ax.set_xlabel('Average tokens per sample (max 96)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test-subject accuracy vs token budget (mean ± sd, 3 person splits)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='center right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'budget_vs_accuracy.png'), dpi=150)


def tiers():
    acc = load('sensor_cut', 'summary.json')['acc']
    ks = [4, 8, 16, 32]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for mod, label in [('depth', 'Depth'), ('mmwave', 'mmWave'), ('lidar', 'LiDAR')]:
        a = acc[mod]['after_projector_pool']
        ax.plot(ks, [100 * a[str(k)] for k in ks], marker='o', label=label)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel('Tokens kept (average pooling after the projector)')
    ax.set_ylabel('Single-sensor accuracy (%)')
    ax.set_title('Token tiers, one sensor at a time (16,000-frame subset)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'tiers_single_sensor.png'), dpi=150)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    budget_curve()
    tiers()
