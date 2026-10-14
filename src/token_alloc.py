"""Run the frozen X-Fi with a per-modality token budget.

X-Fi turns every active modality into 32 tokens (B x 32 x 512) and fuses them.  Here each
modality's 32 projected tokens are average-pooled to k in {4, 8, 16, 32} (k = 0 switches the
modality off).  With every active modality at 32 this is exactly X-Fi's own forward pass.
"""
import itertools

import torch

TIERS = [0, 4, 8, 16, 32]
SLOT_MODS = ['depth', 'mmwave', 'lidar']          # X-Fi slot order after rgb: [rgb, depth, mmwave, lidar]
ALLOCS = [a for a in itertools.product(TIERS, repeat=3) if sum(a) > 0]   # (depth, mmwave, lidar) token counts


def alloc_name(a):
    return f'D{a[0]}-M{a[1]}-L{a[2]}'


POOL_MODE = 'mean'   # 'mean': average contiguous groups; 'uniform': keep every 32/k-th token;
                     # 'norm': keep the k tokens with the largest L2 norm (original order kept)


def pool_tokens(x, k):
    """B x 32 x C -> B x k x C."""
    if k == 32:
        return x
    b, n, c = x.shape
    if POOL_MODE == 'mean':
        return x.reshape(b, k, n // k, c).mean(2)
    if POOL_MODE == 'uniform':
        return x[:, ::n // k]
    if POOL_MODE == 'norm':
        idx = x.norm(dim=2).topk(k, dim=1).indices.sort(dim=1).values
        return x.gather(1, idx[:, :, None].expand(-1, -1, c))
    raise ValueError(POOL_MODE)


def fusion_forward(xf, feature_list, ml):
    """X_Fusion.forward with the per-modality token lists passed in (they may differ in length)."""
    kv_list, idx = [], 0
    for i, kv_layer in enumerate(xf.kv_layers):
        if ml[i]:
            kv_list.append(kv_layer(feature_list[idx]))
            idx += 1
    emb = xf.cross_modal_transformer(torch.cat(feature_list, dim=1), ml)
    for _ in range(xf.depth):
        emb = xf.cross_attention_transformer(emb, kv_list, ml)
        emb = xf.cross_modal_transformer(emb, ml)
    return xf.classification_head(emb)


def precompute(model, feats, lidar_points):
    """Per-batch work shared by all allocations: per-modality projection to 32 tokens and the
    lidar positional encoding.  Farthest-point sampling is greedy from one random start, so with a
    fixed seed the 32/64-point samples are prefixes of the 96-point one; one FPS call covers all."""
    from X_Fi import selective_pos_enc
    lp = model.linear_projector
    heads = {'depth': lp.depth_linear_projection, 'mmwave': lp.mmwave_linear_projection,
             'lidar': lp.lidar_linear_projection}
    proj = {m: heads[m](feats[m].permute(0, 2, 1)).permute(0, 2, 1) for m in SLOT_MODS}   # B x 32 x 512
    new_xyz = selective_pos_enc(lidar_points, 96)
    pos = lp.pos_enc_layer(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)                     # B x 96 x 512
    return proj, pos


def run_alloc_fast(model, pre, alloc):
    """Same result as run_alloc (given the same FPS draw), without recomputing shared parts."""
    proj, pos = pre
    ml = [False] + [k > 0 for k in alloc]
    active = [(m, k) for m, k in zip(SLOT_MODS, alloc) if k > 0]
    x = torch.cat([proj[m] for m, _ in active], dim=1)
    if alloc[2] > 0:
        x = x + pos[:, :x.shape[1]]
    pooled = [pool_tokens(c, k) for c, (_, k) in zip(x.split(32, dim=1), active)]
    return fusion_forward(model.X_Fusion_block, pooled, ml)


def run_alloc(model, feats, lidar_points, alloc):
    """feats: dict mod -> backbone feature (from model.feature_extractor); alloc: (kD, kM, kL)."""
    ml = [False] + [k > 0 for k in alloc]
    active = [(m, k) for m, k in zip(SLOT_MODS, alloc) if k > 0]
    # projection + lidar positional encoding exactly as X-Fi does it, on the full 32 tokens
    proj = model.linear_projector([feats[m] for m, _ in active], lidar_points, ml)   # B x 32n x 512
    chunks = proj.split(32, dim=1)
    pooled = [pool_tokens(c, k) for c, (_, k) in zip(chunks, active)]
    return fusion_forward(model.X_Fusion_block, pooled, ml)
