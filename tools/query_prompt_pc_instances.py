#!/usr/bin/env python3
"""
query_prompt_pc_instances.py (CloudCompare-safe)

- Loads map.pt with:
    feats_xyz  (N,3)
    feats_feats (N,C)
    instance_id (N,)   # optional, but required for instance-colored output
- Computes similarity to a text prompt
- Writes TWO PLYs:
    1) --out_ply_red: prompted class points lit in red (intensity by score or hard red),
       background black (if --bg_black).
    2) --out_ply_instances: prompted class points colored by instance_id (different color per instance),
       background black.

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
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def make_palette(num_colors: int, seed: int = 0) -> np.ndarray:
    """
    Returns (num_colors, 3) uint8 palette.
    Index 0 is black.
    """
    rng = np.random.default_rng(seed)
    pal = rng.integers(32, 256, size=(num_colors, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)
    return pal


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_pt", required=True)
    ap.add_argument("--prompt", required=True)

    ap.add_argument("--out_ply_red", required=True, help="output PLY where selected points are red, bg black")
    ap.add_argument("--out_ply_instances", required=True, help="output PLY where selected points colored by instance_id, bg black")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--th", type=float, default=0.15, help="similarity threshold (ignored if --topk>0)")
    ap.add_argument("--topk", type=int, default=0, help="if >0, select top-K by score")
    ap.add_argument("--bg_black", action="store_true", help="keep all points, non-selected are black")
    ap.add_argument("--hard_red", action="store_true", help="selected points are (255,0,0) in red-output ply")

    # instance coloring options
    ap.add_argument("--seed", type=int, default=0, help="palette seed for instance colors")
    ap.add_argument("--min_instance_points", type=int, default=1, help="drop instances with fewer selected points than this")
    ap.add_argument("--unknown_color", type=int, nargs=3, default=[255, 255, 255], help="RGB for instance_id==0 within selected points")

    # encoder params
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    d = torch.load(args.map_pt, map_location="cpu")
    feats_xyz_cpu = d["feats_xyz"]
    feats = d["feats_feats"].to(args.device)

    if "instance_id" in d:
        inst_id_cpu = d["instance_id"]
        if not isinstance(inst_id_cpu, torch.Tensor):
            inst_id_cpu = torch.tensor(inst_id_cpu)
        inst_id_cpu = inst_id_cpu.detach().cpu().numpy().astype(np.int32)
    else:
        inst_id_cpu = None

    if not isinstance(feats_xyz_cpu, torch.Tensor):
        feats_xyz_cpu = torch.tensor(feats_xyz_cpu)
    xyz_cpu_np = feats_xyz_cpu.detach().cpu().numpy()

    N = xyz_cpu_np.shape[0]
    if feats.shape[0] != N:
        raise RuntimeError(f"map.pt inconsistent: feats_xyz has {N} points but feats_feats has {feats.shape[0]}")

    if inst_id_cpu is None:
        raise RuntimeError("map.pt does not contain instance_id. Build the map with instances first.")

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
        _, sel_idx = torch.topk(scores, k=k, largest=True)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[sel_idx] = True
    else:
        mask = scores >= float(args.th)

    mask_np = mask.detach().cpu().numpy()
    sel_n = int(mask_np.sum())
    if sel_n == 0:
        raise RuntimeError("No points selected. Lower --th or use --topk.")

    # ---------------------------
    # Output 1: all selected in red (bg black optional)
    # ---------------------------
    if args.bg_black:
        rgb_red = np.zeros((N, 3), dtype=np.uint8)
        if args.hard_red:
            rgb_red[mask_np, 0] = 255
        else:
            s = scores[mask]
            s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
            red = (s01 * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            rgb_red[mask_np, 0] = red
        write_ply_xyzrgb_ascii(args.out_ply_red, xyz_cpu_np, rgb_red)
    else:
        xyz_sel = xyz_cpu_np[mask_np]
        if args.hard_red:
            rgb_sel = np.zeros((xyz_sel.shape[0], 3), dtype=np.uint8)
            rgb_sel[:, 0] = 255
        else:
            s = scores[mask]
            s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
            red = (s01 * 255.0).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()
            rgb_sel = np.zeros((xyz_sel.shape[0], 3), dtype=np.uint8)
            rgb_sel[:, 0] = red
        write_ply_xyzrgb_ascii(args.out_ply_red, xyz_sel, rgb_sel)

    # ---------------------------
    # Output 2: instance-colored (bg black optional)
    # ---------------------------
    sel_inst = inst_id_cpu[mask_np]  # instance ids of selected points
    # optionally drop tiny instances among selected points
    if args.min_instance_points > 1:
        ids, counts = np.unique(sel_inst, return_counts=True)
        keep_ids = set(ids[counts >= args.min_instance_points].tolist())
        mask2_np = mask_np.copy()
        # unselect points whose instance is too small
        for i in np.where(mask_np)[0]:
            if inst_id_cpu[i] not in keep_ids:
                mask2_np[i] = False
        mask_np2 = mask2_np
    else:
        mask_np2 = mask_np

    sel_n2 = int(mask_np2.sum())
    if sel_n2 == 0:
        raise RuntimeError("After --min_instance_points filtering, no points remain. Lower it.")

    max_id = int(inst_id_cpu.max()) if inst_id_cpu.size > 0 else 0
    palette = make_palette(max_id + 1, seed=args.seed)
    unknown_color = np.array(args.unknown_color, dtype=np.uint8)

    if args.bg_black:
        rgb_inst = np.zeros((N, 3), dtype=np.uint8)
        sel_indices = np.where(mask_np2)[0]
        sel_ids = inst_id_cpu[sel_indices]
        # assign colors by instance id
        for idx, iid in zip(sel_indices, sel_ids):
            if iid <= 0:
                rgb_inst[idx] = unknown_color
            else:
                rgb_inst[idx] = palette[iid]
        write_ply_xyzrgb_ascii(args.out_ply_instances, xyz_cpu_np, rgb_inst)
    else:
        sel_indices = np.where(mask_np2)[0]
        xyz_sel = xyz_cpu_np[sel_indices]
        rgb_sel = np.zeros((xyz_sel.shape[0], 3), dtype=np.uint8)
        sel_ids = inst_id_cpu[sel_indices]
        for j, iid in enumerate(sel_ids):
            if iid <= 0:
                rgb_sel[j] = unknown_color
            else:
                rgb_sel[j] = palette[iid]
        write_ply_xyzrgb_ascii(args.out_ply_instances, xyz_sel, rgb_sel)

    print(f"[OK] wrote {args.out_ply_red} (red) and {args.out_ply_instances} (instances)")
    print(f"[INFO] selected={sel_n}/{N}, instance-colored-selected={sel_n2}/{N}")


if __name__ == "__main__":
    main()



#     python tools/query_prompt_pc_instances.py \
#   --map_pt radseg_prompt_vis/bcd2436daf_inst/map.pt \
#   --prompt "chair" \
#   --out_ply_red radseg_prompt_vis/bcd2436daf_inst/chair_all_red.ply \
#   --out_ply_instances radseg_prompt_vis/bcd2436daf_inst/chair_instances_color.ply \
#   --topk 10000 \
#   --bg_black \
#   --device cuda \
#   --unknown_color 0 0 0
#   --amp