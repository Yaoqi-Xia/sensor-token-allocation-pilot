"""Check that shared feature extraction gives the same logits as calling model(...) per combination."""
import pilot_paths  # noqa: E402
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syn_DI_dataset import MetaFi_Database, Domain_Invariant_Dataset, decode_config  # noqa: E402
from X_Fi import X_Fi  # noqa: E402
from eval_combos import COMBOS, SLOT, MODS, collate  # noqa: E402

with open('config.yaml') as fd:
    cfg = yaml.load(fd, Loader=yaml.FullLoader)
cfg['modality'] = MODS
val_cfg = decode_config(cfg)['val_dataset']
val_cfg['data_form'] = {'S01': val_cfg['data_form']['S01'][:3]}
ds = Domain_Invariant_Dataset(MetaFi_Database(pilot_paths.MMFI), **val_cfg)
depth, mmwave, lidar, y, _ = next(iter(DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)))
device = torch.device('cuda:0')
model = X_Fi(model_depth=2).to(device)
model.load_state_dict(torch.load('./pre-trained_weights/mmfi_har_checkpoint.pt', map_location=device))
model.eval()
dummy = torch.zeros(1, 3, 1, 1, device=device)
depth, mmwave, lidar = depth.to(device), mmwave.to(device), lidar.to(device)
with torch.no_grad():
    feats = dict(zip(['depth', 'mmwave', 'lidar'], model.feature_extractor(dummy, depth, mmwave, lidar, [False, True, True, True])))
    for c in COMBOS:
        ml = [False] * 4
        for mod in c:
            ml[SLOT[mod]] = True
        a = model(dummy, depth, mmwave, lidar, ml)
        b = model.X_Fusion_block(model.linear_projector([feats[m] for m in ['depth', 'mmwave', 'lidar'] if m in c], lidar, ml), ml)
        a2 = model(dummy, depth, mmwave, lidar, ml)
        print(f'{"+".join(c):20s} max|direct-shared|={(a - b).abs().max():.4f}  max|direct-direct|={(a - a2).abs().max():.4f}  '
              f'argmax agree={(a.argmax(1) == b.argmax(1)).float().mean():.2f}  acc direct={(a.argmax(1).cpu() == y).float().mean():.2f}')
