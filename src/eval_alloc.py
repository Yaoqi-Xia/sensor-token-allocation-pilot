"""Sweep every token allocation (depth, mmwave, lidar) in {0,4,8,16,32}^3 on the X-Fi validation
data and save per-sample logits.  Run from X-Fi/MMFi_HAR.

LiDAR makes X-Fi stochastic (random farthest-point sampling).  Within one batch every allocation
uses the same backbone features and the same random seed, so differences between allocations
are not drowned in sampling noise; --repeats re-runs everything with new seeds.
"""
import pilot_paths  # noqa: E402
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syn_DI_dataset import MetaFi_Database, Domain_Invariant_Dataset, decode_config  # noqa: E402
from X_Fi import X_Fi  # noqa: E402
from eval_combos import ACTIONS, MODS  # noqa: E402
import token_alloc  # noqa: E402
from token_alloc import ALLOCS, alloc_name, precompute, run_alloc_fast  # noqa: E402


def collate(batch):
    labels = torch.LongTensor([ACTIONS[t['action']] for t in batch])
    depth = torch.FloatTensor(np.array([t['input_depth'] for t in batch])).permute(0, 3, 1, 2)
    mmwave = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t['input_mmwave']) for t in batch]).permute(1, 0, 2)
    lidar = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t['input_lidar'].copy()) for t in batch]).permute(1, 0, 2)
    meta = [(t['subject'], t['action'], t['idx']) for t in batch]
    # cheap per-sample quality cues for the router
    quality = np.array([[t['input_lidar'].shape[0], t['input_mmwave'].shape[0],
                         float((t['input_depth'][..., 0] > 0).mean())] for t in batch], dtype=np.float32)
    return depth, mmwave, lidar, labels, meta, quality


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=pilot_paths.MMFI)
    p.add_argument('--pt_weights', default='./pre-trained_weights/mmfi_har_checkpoint.pt')
    p.add_argument('--out', default=os.path.join(pilot_paths.RESULTS, 'alloc_logits.npz'))
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--chunk', default='', help="e.g. '0/2' for the first half of the samples")
    p.add_argument('--pool', default='mean', choices=['mean', 'uniform', 'norm'], help='how 32 tokens shrink to k')
    args = p.parse_args()
    token_alloc.POOL_MODE = args.pool

    with open('config.yaml') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    cfg['modality'] = MODS
    val_cfg = decode_config(cfg)['val_dataset']
    form = {}
    for s, acts in val_cfg['data_form'].items():
        env = f'E0{(int(s[1:]) - 1) // 10 + 1}'
        for a in acts:
            d = os.path.join(args.dataset, env, s, a)
            if all(os.path.exists(os.path.join(d, f'.done_{m}')) for m in ['depth', 'lidar', 'mmwave', 'gt']):
                form.setdefault(s, []).append(a)
    val_cfg['data_form'] = form
    ds = Domain_Invariant_Dataset(MetaFi_Database(args.dataset), **val_cfg)
    if args.chunk:   # 'i/k': i-th of k contiguous sample ranges, so long sweeps can run as separate jobs
        i, k = map(int, args.chunk.split('/'))
        bounds = np.linspace(0, len(ds), k + 1).astype(int)
        ds = torch.utils.data.Subset(ds, range(bounds[i], bounds[i + 1]))
        print(f'chunk {i}/{k}: samples {bounds[i]}..{bounds[i + 1]}')
    n = len(ds) if not args.limit else min(len(ds), args.limit * args.batch_size)
    print(f'samples: {n}, allocations: {len(ALLOCS)}, repeats: {args.repeats}')

    device = torch.device('cuda:0')
    torch.manual_seed(3407)
    model = X_Fi(model_depth=2).to(device)
    model.load_state_dict(torch.load(args.pt_weights, map_location=device))
    model.eval()
    dummy_rgb = torch.zeros(1, 3, 1, 1, device=device)

    logits = np.zeros((args.repeats, n, len(ALLOCS), 27), dtype=np.float16)
    labels = np.zeros(n, dtype=np.int64)
    quality = np.zeros((n, 3), dtype=np.float32)
    meta = []
    t0 = time.time()
    for r in range(args.repeats):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.workers)
        pos = 0
        with torch.no_grad():
            for b, (depth, mmwave, lidar, y, m, q) in enumerate(tqdm(loader, mininterval=60)):
                if pos >= n:
                    break
                depth, mmwave, lidar = depth.to(device), mmwave.to(device), lidar.to(device)
                seed = 1000003 * r + b
                torch.manual_seed(seed)
                feats = dict(zip(['depth', 'mmwave', 'lidar'], model.feature_extractor(
                    dummy_rgb, depth, mmwave, lidar, [False, True, True, True])))
                bs = y.shape[0]
                out = torch.empty(bs, len(ALLOCS), 27, device=device)
                torch.manual_seed(seed)   # one FPS draw shared by every allocation
                pre = precompute(model, feats, lidar)
                for j, a in enumerate(ALLOCS):
                    out[:, j] = run_alloc_fast(model, pre, a)
                logits[r, pos:pos + bs] = out.half().cpu().numpy()
                if r == 0:
                    labels[pos:pos + bs] = y.numpy()
                    quality[pos:pos + bs] = q
                    meta.extend(m)
                pos += bs
    print(f'done in {(time.time() - t0) / 60:.1f} min')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, logits=logits, labels=labels, quality=quality,
             allocs=np.array(ALLOCS), alloc_names=np.array([alloc_name(a) for a in ALLOCS]),
             subject=np.array([x[0] for x in meta]), action=np.array([x[1] for x in meta]),
             frame=np.array([x[2] for x in meta]))
    acc = (logits.astype(np.float32).mean(0).argmax(-1) == labels[:, None]).mean(0)
    for a, ac in zip(ALLOCS, acc):
        if all(k in (0, 32) for k in a) or len(set(a)) == 1:
            print(f'{alloc_name(a):14s} acc={ac * 100:.2f}%')
    print('saved', args.out)


if __name__ == '__main__':
    main()
