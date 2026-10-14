"""Per-sample router inputs that the sweep did not save:
 - each sensor's encoder output averaged over its tokens (depth 49x512, mmwave/lidar 32x512 -> 512);
 - attention share per sensor inside X-Fi's fusion when every sensor gets 4 tokens (probe for the
   attention-allocation baseline).
Same sample order as eval_alloc.py.  Run from X-Fi/MMFi_HAR."""
import pilot_paths  # noqa: E402
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syn_DI_dataset import MetaFi_Database, Domain_Invariant_Dataset, decode_config  # noqa: E402
from X_Fi import X_Fi  # noqa: E402
from eval_combos import MODS  # noqa: E402
from eval_alloc import collate  # noqa: E402
from token_alloc import SLOT_MODS, precompute, run_alloc_fast  # noqa: E402

PROBE = (4, 4, 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=pilot_paths.MMFI)
    p.add_argument('--pt_weights', default='./pre-trained_weights/mmfi_har_checkpoint.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--chunk', default='')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()

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
    if args.chunk:
        i, k = map(int, args.chunk.split('/'))
        bounds = np.linspace(0, len(ds), k + 1).astype(int)
        ds = torch.utils.data.Subset(ds, range(bounds[i], bounds[i + 1]))
    n = len(ds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.workers)

    device = torch.device('cuda:0')
    torch.manual_seed(3407)
    model = X_Fi(model_depth=2).to(device)
    model.load_state_dict(torch.load(args.pt_weights, map_location=device))
    model.eval()
    dummy = torch.zeros(1, 3, 1, 1, device=device)

    # attention received by each key token in the fusion's cross-modal self-attention
    att_calls = []
    mha = model.X_Fusion_block.cross_modal_transformer.attention

    def hook(_m, _inp, out):
        q, k, _ = out.reshape(out.shape[0], out.shape[1], mha.num_heads, -1, 3).unbind(-1)   # b n h d
        att = torch.softmax(torch.einsum('bqhd,bkhd->bhqk', q, k), -1)
        att_calls.append(att.mean((1, 2)))                                                     # b x n
    mha.qkv.register_forward_hook(hook)

    feats = np.zeros((n, 3, 512), dtype=np.float16)
    att = np.zeros((n, 3), dtype=np.float32)
    probe_logits = np.zeros((n, 27), dtype=np.float32)
    meta, pos = [], 0
    with torch.no_grad():
        for b, (depth, mmwave, lidar, y, m, q) in enumerate(tqdm(loader, mininterval=60)):
            depth, mmwave, lidar = depth.to(device), mmwave.to(device), lidar.to(device)
            torch.manual_seed(b)
            f = dict(zip(['depth', 'mmwave', 'lidar'], model.feature_extractor(
                dummy, depth, mmwave, lidar, [False, True, True, True])))
            bs = y.shape[0]
            feats[pos:pos + bs] = torch.stack([f[mod].mean(1) for mod in SLOT_MODS], 1).half().cpu().numpy()
            att_calls.clear()
            pre = precompute(model, f, lidar)
            probe_logits[pos:pos + bs] = run_alloc_fast(model, pre, PROBE).cpu().numpy()
            shares = []
            for a in att_calls:                    # 1st call: 4 tokens per sensor; later calls: 32 per sensor
                per = a.reshape(bs, 3, -1).sum(2)
                shares.append(per / per.sum(1, keepdim=True))
            att[pos:pos + bs] = torch.stack(shares).mean(0).cpu().numpy()
            meta.extend(m)
            pos += bs
    np.savez(args.out, feats=feats, att=att, probe_logits=probe_logits,
             subject=np.array([x[0] for x in meta]), action=np.array([x[1] for x in meta]),
             frame=np.array([x[2] for x in meta]))
    print('saved', args.out, 'mean attention share (D, M, L):', att.mean(0))


if __name__ == '__main__':
    main()
