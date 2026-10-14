"""Run the released X-Fi MM-Fi HAR checkpoint on every combination of depth / lidar / mmwave
(RGB left out) and save per-sample logits for each combination.

Same validation split, loader and model as X-Fi's validate_all.py; only the modality list and
the RGB input (a dummy tensor, never used because RGB is always switched off) differ.
Run from X-Fi/MMFi_HAR so the backbone paths resolve.
"""
import pilot_paths  # noqa: E402
import argparse
import itertools
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from syn_DI_dataset import MetaFi_Database, Domain_Invariant_Dataset, decode_config  # noqa: E402
from X_Fi import X_Fi  # noqa: E402

MODS = ['depth', 'lidar', 'mmwave']
# X-Fi modality_list order is [rgb, depth, mmwave, lidar]
SLOT = {'depth': 1, 'mmwave': 2, 'lidar': 3}
COMBOS = [c for r in (1, 2, 3) for c in itertools.combinations(MODS, r)]
ACTIONS = {f'A{i:02d}': i - 1 for i in range(1, 28)}


def collate(batch):
    labels = torch.LongTensor([ACTIONS[t['action']] for t in batch])
    depth = torch.FloatTensor(np.array([t['input_depth'] for t in batch])).permute(0, 3, 1, 2)
    mmwave = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t['input_mmwave']) for t in batch]).permute(1, 0, 2)
    lidar = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t['input_lidar'].copy()) for t in batch]).permute(1, 0, 2)
    meta = [(t['subject'], t['action'], t['idx']) for t in batch]
    return depth, mmwave, lidar, labels, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=pilot_paths.MMFI)
    p.add_argument('--pt_weights', default='./pre-trained_weights/mmfi_har_checkpoint.pt')
    p.add_argument('--out', default=os.path.join(pilot_paths.RESULTS, 'logits_all_combos.npz'))
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--limit', type=int, default=0, help='only first N batches (smoke test)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--available_only', action='store_true', help='skip folders not downloaded yet')
    args = p.parse_args()

    with open('config.yaml') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    cfg['modality'] = MODS
    val_cfg = decode_config(cfg)['val_dataset']
    if args.available_only:
        # keep only (subject, action) folders whose download finished
        form, n_all = {}, 0
        for s, acts in val_cfg['data_form'].items():
            env = f'E0{(int(s[1:]) - 1) // 10 + 1}'
            for a in acts:
                n_all += 1
                d = os.path.join(args.dataset, env, s, a)
                if all(os.path.exists(os.path.join(d, f'.done_{m}')) for m in ['depth', 'lidar', 'mmwave', 'gt']):
                    form.setdefault(s, []).append(a)
        print(f'available (subject, action) folders: {sum(map(len, form.values()))}/{n_all}')
        val_cfg['data_form'] = form
    ds =Domain_Invariant_Dataset(MetaFi_Database(args.dataset), **val_cfg)
    print('validation samples:', len(ds))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=args.workers, persistent_workers=args.workers > 0)

    device = torch.device('cuda:0')
    torch.manual_seed(3407)
    model = X_Fi(model_depth=2).to(device)
    model.load_state_dict(torch.load(args.pt_weights, map_location=device))
    model.eval()
    dummy_rgb = torch.zeros(1, 3, 1, 1, device=device)

    logits = {c: [] for c in COMBOS}
    labels, meta = [], []
    with torch.no_grad():
        for b, (depth, mmwave, lidar, y, m) in enumerate(tqdm(loader, mininterval=30)):
            if args.limit and b >= args.limit:
                break
            depth, mmwave, lidar = depth.to(device), mmwave.to(device), lidar.to(device)
            # backbones are frozen and independent per modality, so extract each modality once
            # and reuse it for every combination (same result as calling model(...) per combo)
            feats = dict(zip(['depth', 'mmwave', 'lidar'], model.feature_extractor(
                dummy_rgb, depth, mmwave, lidar, [False, True, True, True])))
            for c in COMBOS:
                ml = [False] * 4
                for mod in c:
                    ml[SLOT[mod]] = True
                flist = [feats[mod] for mod in ['depth', 'mmwave', 'lidar'] if mod in c]  # slot order
                proj = model.linear_projector(flist, lidar, ml)
                logits[c].append(model.X_Fusion_block(proj, ml).float().cpu())
            labels.append(y)
            meta.extend(m)

    labels = torch.cat(labels).numpy()
    names = ['+'.join(c) for c in COMBOS]
    L = np.stack([torch.cat(logits[c]).numpy() for c in COMBOS], axis=1)  # N x 7 x 27
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, logits=L, labels=labels, combos=np.array(names),
                        subject=np.array([x[0] for x in meta]), action=np.array([x[1] for x in meta]),
                        frame=np.array([x[2] for x in meta]))
    acc = (L.argmax(-1) == labels[:, None]).mean(0)
    for n, a in zip(names, acc):
        print(f'{n:22s} acc={a * 100:.2f}%')
    print('saved', args.out)


if __name__ == '__main__':
    main()
