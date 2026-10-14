"""Router version B: a small transformer over typed tokens.
  3 sensor tokens  : tier one-hot, quality cue, PCA-32 encoder summary (zeros while the sensor is off)
  1 prediction token: X-Fi's current top-3 probabilities, top probability, entropy
  history tokens   : one per upgrade so far (which sensor, its new tier, change in top prob / entropy)
Each sensor token outputs the predicted MICI of upgrading that sensor one tier.
Training rows come from trajectories (true-MICI greedy + random upgrade orders) so the history
tokens look like the ones seen at inference.  Same splits / selection protocol as router_v2.py."""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

import analyze_alloc as A
from router_v2 import BUDGETS, LAMS, R, SEL_BUDGETS, pca

OUT = os.path.join(R, 'router_v3')
HMAX = 12
DEV = 'cuda'
CONFIGS = {
    'd64-L2': dict(d=64, layers=2, epochs=4, dropout=0.1),
    'd32-L1': dict(d=32, layers=1, epochs=4, dropout=0.1),
}
STATES = np.array(A.STATES)
TIER_IDX = {t: i for i, t in enumerate(A.TIERS)}


class Data:
    """per-(sample, state) quantities shared by training and inference."""
    def __init__(self, P, quality, Z):
        top = -np.sort(-P, 2)
        self.top3 = top[:, :, :3].astype(np.float32)
        self.maxp = top[:, :, 0].astype(np.float32)
        self.ent = (-(P * np.log(P + 1e-9)).sum(2)).astype(np.float32)
        q = np.stack([np.log1p(quality[:, 2] * 100), np.log1p(quality[:, 1]), np.log1p(quality[:, 0])], 1)
        self.q = q.astype(np.float32)          # per slot: depth, mmwave, lidar
        self.Z = Z.astype(np.float32)          # N x 3 x 32

    def tokens(self, i, s, hist, hlen):
        st = STATES[s]                                                       # n x 3
        on = (st > 0)
        onehot = np.zeros((len(i), 3, 5), np.float32)
        for m in range(3):
            onehot[np.arange(len(i)), m, [TIER_IDX[t] for t in st[:, m]]] = 1
        sens = np.concatenate([onehot, (self.q[i] * on)[:, :, None], self.Z[i] * on[:, :, None]], 2)
        pred = np.concatenate([self.top3[i, s], self.maxp[i, s, None], self.ent[i, s, None]], 1)
        mask = np.arange(HMAX)[None] >= hlen[:, None]                       # True = padding
        return sens, pred, hist, mask


class Net(nn.Module):
    def __init__(self, d, layers, dropout):
        super().__init__()
        self.ls, self.lp, self.lh = nn.Linear(38, d), nn.Linear(5, d), nn.Linear(6, d)
        self.typ = nn.Parameter(torch.randn(4, d) * 0.02)     # depth, mmwave, lidar, prediction
        self.pos = nn.Parameter(torch.randn(HMAX, d) * 0.02)  # history step
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, sens, pred, hist, mask):
        x = torch.cat([self.ls(sens) + self.typ[:3], (self.lp(pred) + self.typ[3])[:, None],
                       self.lh(hist) + self.pos], 1)
        pad = torch.cat([torch.zeros(len(mask), 4, dtype=torch.bool, device=mask.device), mask], 1)
        return self.head(self.enc(x, src_key_padding_mask=pad)[:, :3]).squeeze(-1)


def hist_entry(D, i, s_old, s_new, m):
    e = np.zeros((len(i), 6), np.float32)
    e[np.arange(len(i)), m] = 1
    e[:, 3] = STATES[s_new][np.arange(len(i)), m] / 32
    e[:, 4] = D.maxp[i, s_new] - D.maxp[i, s_old]
    e[:, 5] = D.ent[i, s_new] - D.ent[i, s_old]
    return e


def make_rows(D, g, up, dc, idx, k_random=3, seed=0):
    """states visited along 1 true-greedy + k random full trajectories per sample, with their history."""
    rng = np.random.RandomState(seed)
    I, S, H, L = [], [], [], []
    for k in range(1 + k_random):
        n = len(idx)
        s = np.full(n, A.SID[(0, 0, 0)])
        hist = np.zeros((n, HMAX, 6), np.float32)
        for t in range(HMAX):
            I.append(idx.copy()); S.append(s.copy()); H.append(hist.copy()); L.append(np.full(n, t))
            valid = up[s] >= 0
            if k == 0:
                score = np.where(valid, np.nan_to_num(g[idx, s]) / np.maximum(dc[s], 1), -np.inf)
            else:
                score = np.where(valid, rng.rand(n, 3), -np.inf)
            m = score.argmax(1)
            s_new = up[s, m]
            hist[:, t] = hist_entry(D, idx, s, s_new, m)
            s = s_new
    return np.concatenate(I), np.concatenate(S), np.concatenate(H), np.concatenate(L)


def train(D, g, up, dc, idx, cfg, seed):
    torch.manual_seed(seed)
    i, s, h, hl = make_rows(D, g, up, dc, idx, seed=seed)
    sens, pred, hist, mask = D.tokens(i, s, h, hl)
    T = lambda a, dt=torch.float32: torch.tensor(a, device=DEV, dtype=dt)
    sens, pred, hist, mask = T(sens), T(pred), T(hist), T(mask, torch.bool)
    Y = T(np.clip(np.nan_to_num(g[i, s]), -3, 3))
    M = T(up[s] >= 0)
    mu_s, sd_s = sens.mean((0, 1)), sens.std((0, 1)) + 1e-6
    mu_p, sd_p = pred.mean(0), pred.std(0) + 1e-6
    net = Net(cfg['d'], cfg['layers'], cfg['dropout']).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    n = len(Y)
    for ep in range(cfg['epochs']):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for k in range(0, n, 2048):
            b = perm[k:k + 2048]
            out = net((sens[b] - mu_s) / sd_s, (pred[b] - mu_p) / sd_p, hist[b], mask[b])
            loss = (nn.functional.huber_loss(out, Y[b], reduction='none') * M[b]).sum() / M[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def f(sens_, pred_, hist_, mask_):
        with torch.no_grad():
            out = []
            for k in range(0, len(sens_), 8192):
                out.append(net((T(sens_[k:k + 8192]) - mu_s) / sd_s, (T(pred_[k:k + 8192]) - mu_p) / sd_p,
                               T(hist_[k:k + 8192]), T(mask_[k:k + 8192], torch.bool)).cpu().numpy())
            return np.concatenate(out)
    return f


def greedy(D, P, y, idx, up, dc, f, lam):
    n = len(idx)
    s = np.full(n, A.SID[(0, 0, 0)])
    hist = np.zeros((n, HMAX, 6), np.float32)
    hl = np.zeros(n, int)
    active = np.ones(n, bool)
    for t in range(HMAX):
        a = np.where(active)[0]
        if not len(a):
            break
        sens, pred, h, mask = D.tokens(idx[a], s[a], hist[a], hl[a])
        gp = f(sens, pred, h, mask)
        ratio = np.where(up[s[a]] >= 0, gp / np.maximum(dc[s[a]], 1), -np.inf)
        m = ratio.argmax(1)
        go = (ratio[np.arange(len(a)), m] >= lam) | (s[a] == A.SID[(0, 0, 0)])
        ga = a[go]
        s_new = up[s[ga], m[go]]
        hist[ga, t] = hist_entry(D, idx[ga], s[ga], s_new, m[go])
        hl[ga] += 1
        s[ga] = s_new
        active[a[~go]] = False
    return A.evaluate(P, y, idx, s)


def main(smoke=False):
    os.makedirs(OUT, exist_ok=True)
    P, y, subjects, quality = A.load(sorted(glob.glob(os.path.join(R, 'alloc_full_p*.npz'))))
    F = [np.load(p) for p in sorted(glob.glob(os.path.join(R, 'feats_full_p*.npz')))]
    feats = np.concatenate([f['feats'] for f in F]).astype(np.float32)
    N = len(y)
    nll = -np.log(P[np.arange(N), :, y] + 1e-12)
    up, dc = A.upgrade_table(A.TIERS)
    safe = np.where(up >= 0, up, 0)
    g = np.where(up[None] >= 0, nll[:, :, None] - nll[:, safe], np.nan)
    res, chosen = {}, {}
    seeds = [0] if smoke else range(3)
    for seed in seeds:
        rng = np.random.RandomState(seed)
        subs = np.array(sorted(set(subjects)))
        rng.shuffle(subs)
        te_s, tr_s = set(subs[:len(subs) // 2]), subs[len(subs) // 2:]
        va_s = set(tr_s[:max(2, len(tr_s) // 5)])
        te = np.where([s in te_s for s in subjects])[0]
        tr = np.where([s not in te_s for s in subjects])[0]
        va = np.where([s in va_s for s in subjects])[0]
        fit = np.where([(s not in te_s) and (s not in va_s) for s in subjects])[0]
        if smoke:
            fit, va = fit[::20], va[::20]
        D = Data(P, quality, np.stack([pca(feats[:, m], fit_rows=fit) for m in range(3)], 1))
        best, best_score = None, -1
        for cname, cfg in CONFIGS.items():
            cfg = dict(cfg, epochs=1) if smoke else cfg
            f = train(D, g, up, dc, fit, cfg, seed)
            c = [greedy(D, P, y, va, up, dc, f, l) for l in LAMS]
            score = np.mean([A.envelope_at(c, b) for b in SEL_BUDGETS])
            print(f'seed {seed} {cname}: val {score * 100:.2f}', flush=True)
            if score > best_score:
                best, best_score = cname, score
        chosen[seed] = best
        if smoke:
            return
        D = Data(P, quality, np.stack([pca(feats[:, m], fit_rows=tr) for m in range(3)], 1))
        f = train(D, g, up, dc, tr, CONFIGS[best], seed)
        c = [greedy(D, P, y, te, up, dc, f, l) for l in LAMS]
        res.setdefault('acc', []).append([A.envelope_at(c, b) for b in BUDGETS])
        if seed == 0:
            res['curve_seed0'] = c
        print(f'seed {seed} test:', [round(v * 100, 1) for v in res['acc'][-1]], flush=True)
    acc = np.array(res['acc']) * 100
    table = {f'@{b}': [round(float(acc[:, i].mean()), 1), round(float(acc[:, i].std()), 1)] for i, b in enumerate(BUDGETS)}
    json.dump({'table': table, 'chosen': chosen, 'curve_seed0': res['curve_seed0']},
              open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('版本 B', ' '.join(f'{b}:{m:.1f}±{s:.1f}' for b, (m, s) in table.items()))


if __name__ == '__main__':
    main(smoke='--smoke' in sys.argv)
