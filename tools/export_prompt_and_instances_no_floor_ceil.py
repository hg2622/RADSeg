#!/usr/bin/env python3
"""
Export two point-cloud PLYs from map.pt after removing floor/ceiling points:
1) Prompt-highlight PLY: prompt-matching voxels in red, others in background color.
2) Instance-color PLY: prompt-matching voxels colored by instance_id, others in background color.

This script expects map.pt keys:
- feats_xyz:   (N,3)
- feats_feats: (N,C)
- instance_id: (N,)   # required for instance-colored output

Floor/ceiling removal is done first using text similarity against --remove_prompts.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch

from radseg.radseg import RADSegEncoder


def write_ply_xyzrgb_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (N,3), got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]

    n = xyz.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def make_palette(max_id: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pal = rng.integers(32, 256, size=(max_id + 1, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)
    return pal


def parse_csv_prompts(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def encode_text(enc: RADSegEncoder, prompts: List[str]) -> torch.Tensor:
    if hasattr(enc, "encode_labels"):
        t = enc.encode_labels(prompts)
    else:
        t = enc.encode_prompts(prompts)
    t = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
    return t


def compute_max_sim_scores(
    enc: RADSegEncoder,
    feats: torch.Tensor,
    text_feats: torch.Tensor,
    device: str,
    chunk_size: int,
) -> torch.Tensor:
    """Return max text similarity per point as CPU tensor shape (N,)."""
    n = int(feats.shape[0])
    out = torch.empty((n,), dtype=torch.float32, device="cpu")

    text_feats = text_feats.to(device)
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        f = feats[s:e].to(device, non_blocking=True)
        lang = enc.align_spatial_features_with_language(f.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        lang = lang / (lang.norm(dim=-1, keepdim=True) + 1e-12)
        sim = lang @ text_feats.transpose(0, 1)
        out[s:e] = sim.max(dim=1).values.detach().cpu().float()
        del f, lang, sim

    return out


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_pt", required=True)
    ap.add_argument("--prompt", required=True, help="target class prompt")

    ap.add_argument("--out_ply_prompt_red", required=True)
    ap.add_argument("--out_ply_instance", required=True)

    ap.add_argument("--prompt_th", type=float, default=0.065)
    ap.add_argument("--remove_prompts", default="floor,ceiling")
    ap.add_argument("--remove_th", type=float, default=0.065)
    ap.add_argument("--chunk_size", type=int, default=50000)

    ap.add_argument("--bg_color", type=int, nargs=3, default=[80, 80, 80])
    ap.add_argument("--unknown_instance_color", type=int, nargs=3, default=[255, 255, 255])
    ap.add_argument("--palette_seed", type=int, default=0)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_ply_prompt_red) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ply_instance) or ".", exist_ok=True)

    d = torch.load(args.map_pt, map_location="cpu", weights_only=False)
    feats_xyz = d["feats_xyz"]
    feats = d["feats_feats"]
    inst = d.get("instance_id", None)

    if inst is None:
        raise RuntimeError("map.pt missing 'instance_id'. Build an instance-aware map first.")

    if not isinstance(feats_xyz, torch.Tensor):
        feats_xyz = torch.tensor(feats_xyz)
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)
    if not isinstance(inst, torch.Tensor):
        inst = torch.tensor(inst)

    xyz_np = feats_xyz.detach().cpu().numpy().astype(np.float32)
    inst_np = inst.detach().cpu().numpy().astype(np.int32)

    n = int(xyz_np.shape[0])
    if int(feats.shape[0]) != n or int(inst_np.shape[0]) != n:
        raise RuntimeError("map.pt inconsistent shapes among feats_xyz / feats_feats / instance_id")

    enc = RADSegEncoder(
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=args.device,
        amp=args.amp,
        compile=args.compile,
        predict=False,
        sam_refinement=False,
    )

    remove_prompts = parse_csv_prompts(args.remove_prompts)
    if len(remove_prompts) > 0:
        remove_text = encode_text(enc, remove_prompts)
        remove_scores = compute_max_sim_scores(
            enc=enc,
            feats=feats,
            text_feats=remove_text,
            device=args.device,
            chunk_size=max(1, int(args.chunk_size)),
        )
        keep_mask = (remove_scores < float(args.remove_th)).cpu().numpy()
    else:
        keep_mask = np.ones((n,), dtype=bool)

    target_text = encode_text(enc, [args.prompt])
    target_scores = compute_max_sim_scores(
        enc=enc,
        feats=feats,
        text_feats=target_text,
        device=args.device,
        chunk_size=max(1, int(args.chunk_size)),
    )
    prompt_mask = ((target_scores >= float(args.prompt_th)).cpu().numpy()) & keep_mask

    kept_n = int(keep_mask.sum())
    sel_n = int(prompt_mask.sum())
    if kept_n == 0:
        raise RuntimeError("All points removed by floor/ceiling filtering. Lower --remove_th.")
    if sel_n == 0:
        raise RuntimeError("No prompt-matching points remain. Lower --prompt_th.")

    xyz_keep = xyz_np[keep_mask]
    inst_keep = inst_np[keep_mask]
    sel_keep = prompt_mask[keep_mask]

    bg = np.array(args.bg_color, dtype=np.uint8)

    # PLY 1: prompt in red, non-prompt in background color.
    rgb_red = np.tile(bg[None, :], (xyz_keep.shape[0], 1))
    rgb_red[sel_keep] = np.array([255, 0, 0], dtype=np.uint8)
    write_ply_xyzrgb_ascii(args.out_ply_prompt_red, xyz_keep, rgb_red)

    # PLY 2: prompt points colored by instance id, non-prompt in background color.
    rgb_inst = np.tile(bg[None, :], (xyz_keep.shape[0], 1))
    max_id = int(max(0, inst_keep.max(initial=0)))
    pal = make_palette(max_id=max_id, seed=int(args.palette_seed))
    unk = np.array(args.unknown_instance_color, dtype=np.uint8)

    sel_idx = np.where(sel_keep)[0]
    for i in sel_idx:
        iid = int(inst_keep[i])
        if iid <= 0:
            rgb_inst[i] = unk
        else:
            rgb_inst[i] = pal[iid]

    write_ply_xyzrgb_ascii(args.out_ply_instance, xyz_keep, rgb_inst)

    removed_n = n - kept_n
    print(f"[OK] wrote {args.out_ply_prompt_red}")
    print(f"[OK] wrote {args.out_ply_instance}")
    print(f"[INFO] total={n} removed_floor_ceil={removed_n} kept={kept_n} prompt_selected={sel_n}")
    print(f"[INFO] prompt='{args.prompt}' prompt_th={args.prompt_th}")
    print(f"[INFO] remove_prompts={remove_prompts} remove_th={args.remove_th}")


if __name__ == "__main__":
    main()


# Example:
# python tools/export_prompt_and_instances_no_floor_ceil.py \
#   --map_pt /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_sam_inst/map.pt \
#   --prompt table \
#   --out_ply_prompt_red /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_sam_inst/table_red_no_floorceil.ply \
#   --out_ply_instance /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_sam_inst/table_inst_no_floorceil.ply \
#   --remove_prompts floor,ceiling \
#   --remove_th 0.065 \
#   --prompt_th 0.065 \
#   --device cuda \
#   --amp
