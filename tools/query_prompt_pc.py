#!/usr/bin/env python3
"""
query_prompt_pc.py (CloudCompare-safe)

- Loads map.pt with feats_xyz (N,3) and feats_feats (N,C)
- Computes similarity to a text prompt
- Writes ONE PLY:
    * if --bg_black: all points kept, non-selected points are black
    * else: only selected points are written

Hardening:
- Filters out any rows where xyz has NaN/Inf
- Ensures xyz float32 and rgb uint8
- Ensures header vertex count matches written data
"""

import argparse
import numpy as np
import torch

from radseg.radseg import RADSegEncoder


def write_ply_xyzrgb_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """
    xyz: (N,3) float32/float64
    rgb: (N,3) uint8/int
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (N,3), got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb length mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

    # Force dtypes
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    # Filter non-finite xyz (CloudCompare can choke on NaN/Inf)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]

    n = xyz.shape[0]
    if n == 0:
        raise RuntimeError("No finite points to write after filtering NaN/Inf.")

    with open(path, "w") as f:
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
        # Avoid scientific notation issues by formatting floats explicitly
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_pt", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--th", type=float, default=0.15, help="similarity threshold (ignored if --topk>0)")
    ap.add_argument("--topk", type=int, default=0, help="if >0, select top-K by score")
    ap.add_argument("--bg_black", action="store_true", help="keep all points, non-selected are black")
    ap.add_argument("--hard_red", action="store_true", help="selected points are (255,0,0)")

    # encoder params
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    d = torch.load(args.map_pt, map_location="cpu")
    feats_xyz_cpu = d["feats_xyz"]          # keep a CPU copy for writing
    feats = d["feats_feats"].to(args.device)

    if not isinstance(feats_xyz_cpu, torch.Tensor):
        feats_xyz_cpu = torch.tensor(feats_xyz_cpu)

    # Some maps might store xyz as double; normalize to float32 later
    N = feats_xyz_cpu.shape[0]
    if feats.shape[0] != N:
        raise RuntimeError(f"map.pt inconsistent: feats_xyz has {N} points but feats_feats has {feats.shape[0]}")

    enc = RADSegEncoder(
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=args.device,
        amp=args.amp,
        compile=args.compile,
        predict=False,
        sam_refinement=False,
    )

    # Text embedding
    if hasattr(enc, "encode_labels"):
        text = enc.encode_labels([args.prompt])
    else:
        text = enc.encode_prompts([args.prompt])
    text = text / (text.norm(dim=-1, keepdim=True) + 1e-12)

    # Align voxel feats -> language space
    lang_feats = enc.align_spatial_features_with_language(
        feats.unsqueeze(-1).unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)
    lang_feats = lang_feats / (lang_feats.norm(dim=-1, keepdim=True) + 1e-12)

    scores = (lang_feats @ text[0].unsqueeze(-1)).squeeze(-1)  # (N,)

    # Select
    if args.topk > 0:
        k = min(int(args.topk), int(scores.numel()))
        sel_scores, sel_idx = torch.topk(scores, k=k, largest=True)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[sel_idx] = True
    else:
        mask = scores >= float(args.th)
        sel_scores = scores[mask]

    sel_n = int(mask.sum().item())
    if sel_n == 0:
        raise RuntimeError("No points selected. Lower --th or use --topk.")

    # Build rgb on CPU for writing
    if args.bg_black:
        rgb = np.zeros((N, 3), dtype=np.uint8)

        if args.hard_red:
            rgb[mask.detach().cpu().numpy(), 0] = 255
        else:
            s = scores[mask]
            s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
            red = (s01 * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            m = mask.detach().cpu().numpy()
            rgb[m, 0] = red

        xyz = feats_xyz_cpu.detach().cpu().numpy()
        write_ply_xyzrgb_ascii(args.out_ply, xyz, rgb)
    else:
        m = mask.detach().cpu().numpy()
        xyz = feats_xyz_cpu.detach().cpu().numpy()[m]

        if args.hard_red:
            rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
            rgb[:, 0] = 255
        else:
            s = scores[mask]
            s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
            red = (s01 * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
            rgb[:, 0] = red

        write_ply_xyzrgb_ascii(args.out_ply, xyz, rgb)

    print(f"[OK] wrote {args.out_ply}  selected={sel_n}/{N}")


if __name__ == "__main__":
    main()


#     python tools/query_prompt_pc.py \
#   --map_pt radseg_prompt_vis/5a269ba6fe_fs1/map.pt \
#   --prompt "window frame" \
#   --out_ply radseg_prompt_vis/5a269ba6fe_fs1/lit_window_frame.ply \
#   --th 0.1 \
#  --bg_black \
#   --device cuda \
#   --amp