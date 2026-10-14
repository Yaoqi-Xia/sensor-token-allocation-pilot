"""Encoder-cut sensor test (frozen X-Fi, no training).

Cut point: right after each sensor's encoder, before X-Fi's projector
  depth : ResNet18 feature map  -> B x 49 x 512 (7x7 positions)
  mmwave: point transformer     -> B x 32 x 512
  lidar : point transformer     -> B x 32 x 512
Keep k evenly spaced encoder features (k in 4/8/16/32, plus the full count) and fill the removed
positions with the nearest kept feature (or zeros, as a sanity row), so the projector still gets
its usual input size; projector and fusion run unchanged.  One sensor at a time.
LiDAR randomness (farthest-point sampling start, used in the encoder and in the positional encoding)
is fixed by calling torch.manual_seed(batch index) before the encoder and again before every
projector+fusion call, so all k for a batch see the same draw.
Fixed random subset of the validation set (same frame order as the stored sweep tables).
Run from X-Fi/MMFi_HAR.  Writes only new files under results/sensor_cut/."""
import pilot_paths  # noqa: E402
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syn_DI_dataset import MetaFi_Database, Domain_Invariant_Dataset, decode_config  # noqa: E402
from X_Fi import X_Fi  # noqa: E402
from eval_combos import MODS  # noqa: E402
from eval_alloc import collate  # noqa: E402

OUT = os.path.join(pilot_paths.RESULTS, 'sensor_cut')
N_SUB = int(os.environ.get('N_SUB', 16000))
SENSORS = {'depth': (1, 49), 'mmwave': (2, 32), 'lidar': (3, 32)}      # slot in modality_list, positions
KS = [4, 8, 16, 32]


def keep_index(n, k):
    keep = np.unique(np.linspace(0, n - 1, k).round().astype(int))
    nearest = keep[np.abs(np.arange(n)[:, None] - keep[None]).argmin(1)]   # nearest kept index per position
    return keep, nearest


def main():
    os.makedirs(OUT, exist_ok=True)
    with open('config.yaml') as fd:
        cfg = yaml.load(fd, Loader=yaml.FullLoader)
    cfg['modality'] = MODS
    val_cfg = decode_config(cfg)['val_dataset']
    ds = Domain_Invariant_Dataset(MetaFi_Database(pilot_paths.MMFI), **val_cfg)
    sub = np.sort(np.random.RandomState(0).choice(len(ds), N_SUB, replace=False))
    loader = DataLoader(Subset(ds, sub), batch_size=16, shuffle=False, collate_fn=collate, num_workers=3)

    device = torch.device('cuda:0')
    torch.manual_seed(3407)
    model = X_Fi(model_depth=2).to(device)
    model.load_state_dict(torch.load('./pre-trained_weights/mmfi_har_checkpoint.pt', map_location=device))
    model.eval()
    dummy = torch.zeros(1, 3, 1, 1, device=device)

    variants = {}
    for s, (slot, n) in SENSORS.items():
        for k in KS + ([n] if n not in KS else []):
            variants[(s, k, 'repeat')] = keep_index(n, k)
            if k < n:
                variants[(s, k, 'zero')] = keep_index(n, k)
    names = [f'{s}|{k}|{fill}' for (s, k, fill) in variants]
    preds = np.zeros((N_SUB, len(variants)), np.int16)
    labels = np.zeros(N_SUB, np.int64)
    shapes = {}
    pos, t0 = 0, time.time()
    with torch.no_grad():
        for b, (depth, mmwave, lidar, y, m, q) in enumerate(tqdm(loader, mininterval=60)):
            depth, mmwave, lidar = depth.to(device), mmwave.to(device), lidar.to(device)
            torch.manual_seed(b)
            feats = dict(zip(['depth', 'mmwave', 'lidar'], model.feature_extractor(
                dummy, depth, mmwave, lidar, [False, True, True, True])))
            if b == 0:
                shapes = {s: tuple(f.shape[1:]) for s, f in feats.items()}
            bs = y.shape[0]
            for j, ((s, k, fill), (keep, nearest)) in enumerate(variants.items()):
                x = feats[s]
                if fill == 'repeat':
                    xc = x[:, torch.as_tensor(nearest, device=device)]
                else:
                    xc = torch.zeros_like(x)
                    kt = torch.as_tensor(keep, device=device)
                    xc[:, kt] = x[:, kt]
                ml = [False] * 4
                ml[SENSORS[s][0]] = True
                torch.manual_seed(b)
                proj = model.linear_projector([xc], lidar, ml)
                preds[pos:pos + bs, j] = model.X_Fusion_block(proj, ml).argmax(1).cpu().numpy()
            labels[pos:pos + bs] = y.numpy()
            pos += bs
    np.savez(os.path.join(OUT, 'sensor_cut_preds.npz'), preds=preds, labels=labels, names=np.array(names), subset=sub)
    print('encoder output shapes:', shapes)
    print(f'done in {(time.time() - t0) / 60:.1f} min')
    for j, nm in enumerate(names):
        print(f'{nm:20s} acc={(preds[:, j] == labels).mean() * 100:.2f}')


if __name__ == '__main__':
    main()
