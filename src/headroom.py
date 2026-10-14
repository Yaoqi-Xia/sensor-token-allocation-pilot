"""Headroom check: how much better is picking the best modality combination per sample (oracle)
than using one fixed combination for every sample?"""
import os
import pilot_paths  # noqa: E402
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(pilot_paths.RESULTS, 'logits_all_combos.npz')
d = np.load(path)
L, y, names = d['logits'], d['labels'], list(d['combos'])
N, C, K = L.shape
logp = L - np.log(np.exp(L - L.max(-1, keepdims=True)).sum(-1, keepdims=True)) - L.max(-1, keepdims=True)
nll = -logp[np.arange(N), :, y]            # N x C, -ln p(correct class)
correct = L.argmax(-1) == y[:, None]       # N x C
ntok = np.array([32 * len(n.split('+')) for n in names])

res = {'n_samples': int(N), 'per_combo': {}}
for j, n in enumerate(names):
    res['per_combo'][n] = {'acc': float(correct[:, j].mean()), 'nll': float(nll[:, j].mean()), 'tokens': int(ntok[j])}
best = int(np.argmax(correct.mean(0)))
res['best_fixed'] = {'combo': names[best], 'acc': float(correct[:, best].mean()), 'nll': float(nll[:, best].mean())}

# oracle 1: per sample, any combination gets it right
res['oracle_acc'] = float(correct.any(1).mean())
# oracle 2: per sample, combination with the lowest -ln p(correct)
j_star = nll.argmin(1)
res['oracle_nll'] = float(nll.min(1).mean())
res['oracle_nll_pick_acc'] = float(correct[np.arange(N), j_star].mean())
res['oracle_pick_counts'] = {names[j]: int((j_star == j).sum()) for j in range(C)}
# share of samples where best fixed is wrong but some other combo is right
res['rescued_by_other_combo'] = float((~correct[:, best] & correct.any(1)).mean())
# label-free, realizable selector: most confident combination
conf = np.exp(logp.max(-1))
j_conf = conf.argmax(1)
res['max_confidence_selector_acc'] = float(correct[np.arange(N), j_conf].mean())
# cost view: cheapest correct combination per sample (all three if none correct)
order = np.argsort(ntok, kind='stable')
tok = np.full(N, ntok.max())
for i in range(N):
    for j in order:
        if correct[i, j]:
            tok[i] = ntok[j]
            break
res['oracle_cheapest_correct_avg_tokens'] = float(tok.mean())
res['oracle_cheapest_correct_acc'] = res['oracle_acc']

print(json.dumps(res, indent=2))
out = path.replace('.npz', '_headroom.json')
with open(out, 'w') as f:
    json.dump(res, f, indent=2)
print('saved', out)
