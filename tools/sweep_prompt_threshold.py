#!/usr/bin/env python3
"""
Sweep prompt similarity thresholds on a saved RADSeg 3D map and plot
threshold vs number of selected points.

Example:
  python tools/sweep_prompt_threshold.py \
    --map_pt /path/to/map.pt \
    --out_dir /path/to/out \
    --prompt chair \
    --th_min -0.1 \
    --th_max 0.5 \
    --th_steps 61 \
    --device cuda \
    --amp
"""

import argparse
import os
import re

import numpy as np
import torch

from radseg.radseg import RADSegEncoder


FIXED_THRESHOLDS = [0.05, 0.06,0.065, 0.07, 0.075, 0.08]


def write_ply_xyzrgb_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (N,3), got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb length mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]

    n = xyz.shape[0]
    if n == 0:
        raise RuntimeError("No finite points to write after filtering NaN/Inf.")

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


def _safe_name(txt: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", txt.strip())
    return s if s else "prompt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_pt", required=True, help="Path to map.pt from build_map_one_scene.py")
    ap.add_argument("--out_dir", required=True, help="Directory to save plot + CSV")
    ap.add_argument("--prompt", default="chair", help="Text prompt to query")

    ap.add_argument("--hard_red", action="store_true", help="Use solid red for selected points")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = torch.load(args.map_pt, map_location="cpu")
    feats_xyz = d["feats_xyz"]
    feats = d["feats_feats"]

    if not isinstance(feats_xyz, torch.Tensor):
        feats_xyz = torch.tensor(feats_xyz)
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)

    if feats_xyz.shape[0] != feats.shape[0]:
        raise RuntimeError(
            f"map.pt inconsistent: feats_xyz has {feats_xyz.shape[0]} points, "
            f"feats_feats has {feats.shape[0]} points"
        )

    feats = feats.to(args.device)

    enc = RADSegEncoder(
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=args.device,
        amp=args.amp,
        compile=args.compile,
        predict=False,
        sam_refinement=False,
    )

    if hasattr(enc, "encode_labels"):
        text = enc.encode_labels([args.prompt])
    else:
        text = enc.encode_prompts([args.prompt])
    text = text / (text.norm(dim=-1, keepdim=True) + 1e-12)

    lang_feats = enc.align_spatial_features_with_language(
        feats.unsqueeze(-1).unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)
    lang_feats = lang_feats / (lang_feats.norm(dim=-1, keepdim=True) + 1e-12)

    scores = (lang_feats @ text[0].unsqueeze(-1)).squeeze(-1)

    thresholds = torch.tensor(FIXED_THRESHOLDS, dtype=scores.dtype, device=scores.device)
    counts = [(scores >= t).sum().item() for t in thresholds]

    thresholds_np = thresholds.detach().cpu().numpy()
    counts_np = np.asarray(counts, dtype=np.int64)
    total_pts = int(feats_xyz.shape[0])

    csv_path = os.path.join(args.out_dir, f"threshold_counts_{args.prompt}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("threshold,count,fraction\n")
        for t, c in zip(thresholds_np, counts_np):
            f.write(f"{float(t):.6f},{int(c)},{float(c)/max(total_pts,1):.8f}\n")

    xyz_np = feats_xyz.detach().cpu().numpy().astype(np.float32)
    scores_cpu = scores.detach().cpu()
    mask_dir = os.path.join(args.out_dir, f"prompt_ply_{_safe_name(args.prompt)}")
    os.makedirs(mask_dir, exist_ok=True)

    for t in FIXED_THRESHOLDS:
        mask = scores_cpu >= float(t)
        rgb = np.zeros((total_pts, 3), dtype=np.uint8)
        sel_n = int(mask.sum().item())

        if sel_n > 0:
            if args.hard_red:
                rgb[mask.numpy(), 0] = 255
            else:
                s = scores_cpu[mask]
                s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
                red = (s01 * 255.0).clamp(0, 255).to(torch.uint8).numpy()
                m = mask.numpy()
                rgb[m, 0] = red

        ply_name = f"{_safe_name(args.prompt)}_th{t:g}.ply"
        ply_path = os.path.join(mask_dir, ply_name)
        write_ply_xyzrgb_ascii(ply_path, xyz_np, rgb)
        print(f"[OK] wrote PLY: {ply_path} selected={sel_n}/{total_pts}")

    plot_path = os.path.join(args.out_dir, f"threshold_vs_points_{args.prompt}.png")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(thresholds_np, counts_np, marker="o", markersize=3, linewidth=1.5)
        plt.title(f"Prompt='{args.prompt}': threshold vs lit-up points")
        plt.xlabel("Similarity threshold")
        plt.ylabel("Number of lit-up points")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
        print(f"[OK] wrote plot: {plot_path}")
    except Exception as e:
        print(f"[WARN] matplotlib plot failed: {e}")

    print(f"[OK] wrote CSV: {csv_path}")
    print(f"[INFO] total points: {total_pts}")
    print(f"[INFO] score range: min={scores.min().item():.6f} max={scores.max().item():.6f}")


if __name__ == "__main__":
    main()




# python tools/sweep_prompt_threshold.py \
#   --map_pt /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe/map.pt \
#   --out_dir /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe \
#   --prompt table \
#   --device cuda \
#   --amp