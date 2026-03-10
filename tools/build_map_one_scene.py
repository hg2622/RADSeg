#!/usr/bin/env python3
"""
tools/build_map_one_scene.py

Build a 3D voxel feature map for ONE ScanNet++ SceneVerse singleview scene,
WITHOUT any evaluation/GT.

Input scene structure:
  <data_root>/<scene_id>/
    color/*.png
    depth/*.png
    poses.csv
    intrinsics.txt

Output:
  <out_dir>/map.pt  containing:
    feats_xyz: (N,3) float32
    feats_feats: (N,C) float32
    K: (3,3) float32 (CPU copy for convenience)
"""

import os
import re
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from radseg.radseg import RADSegEncoder
from rayfronts.mapping.semantic_voxel_map import SemanticVoxelMap


# ----------------------------
# Robust intrinsics parsing (ignores '#', words, etc.)
# ----------------------------
def read_intrinsics_any(path: str, img_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    # Infer image size from first non-comment line with 2 tokens (often "W H" or "H W")
    inferred_hw = None
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) == 2:
            try:
                a = int(float(toks[0]))
                b = int(float(toks[1]))
                if a > 0 and b > 0:
                    # assume W H -> (H,W)
                    inferred_hw = (b, a)
                    break
            except Exception:
                pass
    if img_hw is None and inferred_hw is not None:
        img_hw = inferred_hw

    # Grab all numbers
    vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)]
    if len(vals) < 9:
        raise ValueError(f"Unrecognized intrinsics format in {path} (parsed {len(vals)} floats)")

    def _good_K(K: np.ndarray) -> bool:
        if K.shape != (3, 3):
            return False
        # bottom row should be [0,0,1]
        if not (abs(float(K[2, 0])) < 1e-4 and abs(float(K[2, 1])) < 1e-4 and abs(float(K[2, 2] - 1.0)) < 1e-4):
            return False
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if fx <= 0 or fy <= 0:
            return False
        if img_hw is not None:
            H, W = img_hw
            cx, cy = float(K[0, 2]), float(K[1, 2])
            if not (-0.5 * W <= cx <= 1.5 * W and -0.5 * H <= cy <= 1.5 * H):
                return False
        return True

    # Prefer last 3x3 block
    K_last = np.array(vals[-9:], dtype=np.float32).reshape(3, 3)
    if _good_K(K_last):
        return K_last

    # Fallback: scan for plausible fx fy cx cy
    best = None
    best_score = -1e18
    for s in range(0, len(vals) - 4 + 1):
        fx, fy, cx, cy = vals[s:s + 4]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        if not _good_K(K):
            continue
        score = min(fx, fy)
        if img_hw is not None:
            H, W = img_hw
            score -= abs(cx - 0.5 * W) + abs(cy - 0.5 * H)
        if score > best_score:
            best_score = score
            best = K

    if best is not None:
        return best

    raise ValueError(f"Failed to parse a plausible intrinsics K from {path} (parsed {len(vals)} floats)")


# ----------------------------
# Poses parsing (SceneVerse poses.csv)
# Format: image_name,QW,QX,QY,QZ,TX,TY,TZ  (case-insensitive)
# Many exports are w2c; we invert -> c2w.
# ----------------------------
def read_poses_csv_sceneverse(path: str, pose_is_w2c: bool = True) -> Dict[str, np.ndarray]:
    poses: Dict[str, np.ndarray] = {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        cols_l = [c.lower() for c in cols]

        # name column
        name_col = None
        for c in cols:
            if c.lower() in ("image_name", "name", "frame", "image", "filename", "file"):
                name_col = c
                break
        if name_col is None:
            name_col = cols[0]

        def get(row, key_lower: str) -> str:
            for k in row.keys():
                if k.lower() == key_lower:
                    return row[k]
            raise KeyError(key_lower)

        need = ["qw", "qx", "qy", "qz", "tx", "ty", "tz"]
        for k in need:
            if k not in cols_l:
                raise ValueError(f"{path} missing column {k} (has {cols})")

        for row in reader:
            name = str(row[name_col])
            stem = os.path.splitext(os.path.basename(name))[0]

            qw = float(get(row, "qw"))
            qx = float(get(row, "qx"))
            qy = float(get(row, "qy"))
            qz = float(get(row, "qz"))
            tx = float(get(row, "tx"))
            ty = float(get(row, "ty"))
            tz = float(get(row, "tz"))

            # scipy uses [x,y,z,w]
            Rm = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            t = np.array([tx, ty, tz], dtype=np.float32)

            if pose_is_w2c:
                # invert w2c -> c2w
                Rm = Rm.T
                t = (-Rm @ t).astype(np.float32)

            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rm
            T[:3, 3] = t

            # map by several keys
            keys = {name, name.lower(), stem, stem.lower()}
            for k in keys:
                if k:
                    poses[k] = T

    return poses


def load_sceneverse_singleview(
    data_root: str,
    scene: str,
    frame_skip: int,
    rgb_hw: Optional[Tuple[int, int]] = None,
    pose_is_w2c: bool = True,
) -> Tuple[List[Tuple[str, str, np.ndarray]], np.ndarray]:
    scene_root = os.path.join(data_root, scene)
    color_dir = os.path.join(scene_root, "color")
    depth_dir = os.path.join(scene_root, "depth")
    intr_path = os.path.join(scene_root, "intrinsics.txt")
    poses_path = os.path.join(scene_root, "poses.csv")

    for p in [color_dir, depth_dir, intr_path, poses_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    K = read_intrinsics_any(intr_path, img_hw=rgb_hw)
    poses = read_poses_csv_sceneverse(poses_path, pose_is_w2c=pose_is_w2c)

    color_files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(".png")])
    if len(color_files) == 0:
        raise RuntimeError(f"No color pngs in {color_dir}")

    frames: List[Tuple[str, str, np.ndarray]] = []
    for cf in color_files:
        stem = os.path.splitext(cf)[0]
        rgb_path = os.path.join(color_dir, cf)
        depth_path = os.path.join(depth_dir, stem + ".png")
        if not os.path.exists(depth_path):
            continue
        pose = poses.get(stem, poses.get(stem.lower()))
        if pose is None:
            continue
        frames.append((rgb_path, depth_path, pose))

    if len(frames) == 0:
        raise RuntimeError("No paired frames found (need matching stems in color/, depth/, poses.csv)")

    frames = frames[::max(1, frame_skip)]
    return frames, K


def read_depth_m(depth_path: str, depth_scale: float = 1000.0) -> np.ndarray:
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(depth_path)
    if d.dtype == np.uint16:
        depth_m = d.astype(np.float32) / float(depth_scale)
    else:
        depth_m = d.astype(np.float32)
    depth_m[depth_m <= 0] = np.inf
    return depth_m


def render_sam_overlay(rgb: np.ndarray, sam_masks: List[Dict]) -> np.ndarray:
    if len(sam_masks) == 0:
        return rgb.copy()

    out = rgb.copy().astype(np.float32)
    rng = np.random.default_rng(0)
    sorted_masks = sorted(sam_masks, key=lambda m: int(m.get("area", 0)), reverse=True)
    for m in sorted_masks:
        seg = m.get("segmentation", None)
        if seg is None:
            continue
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8).astype(np.float32)
        out[seg] = 0.35 * out[seg] + 0.65 * color

    return out.clip(0, 255).astype(np.uint8)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame_skip", type=int, default=10)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--rgb_h", type=int, default=-1)
    ap.add_argument("--rgb_w", type=int, default=-1)

    # mapper params
    ap.add_argument("--vox_size", type=float, default=0.01)
    ap.add_argument("--max_pts_per_frame", type=int, default=-1)

    # encoder params
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--slide_crop", type=int, default=336)
    ap.add_argument("--slide_stride", type=int, default=112)
    ap.add_argument("--scga_scaling", type=float, default=10.0)
    ap.add_argument("--scra_scaling", type=float, default=10.0)

    # SAM visualization export (independent from RADSeg+ refinement)
    ap.add_argument("--sam_viz_count", type=int, default=10)
    ap.add_argument("--sam_viz_dir", default="sam_seg_vis")
    ap.add_argument("--sam_model_type", default="vit_h")
    ap.add_argument("--sam_ckpt", default="checkpoints/sam_vit_h_4b8939.pth")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)

    rgb_hw = (args.rgb_h, args.rgb_w) if args.rgb_h > 0 and args.rgb_w > 0 else None
    frames, K = load_sceneverse_singleview(
        data_root=args.data_root,
        scene=args.scene,
        frame_skip=args.frame_skip,
        rgb_hw=rgb_hw,
        pose_is_w2c=True,  # matches your verified reference behavior
    )
    print(f"[INFO] scene={args.scene} frames={len(frames)}")

    # Encoder on device
    enc = RADSegEncoder(
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=str(device),
        amp=args.amp,
        compile=args.compile,
        slide_crop=args.slide_crop,
        slide_stride=args.slide_stride,
        scga_scaling=args.scga_scaling,
        scra_scaling=args.scra_scaling,
        predict=False,
        sam_refinement=False,
    )

    # IMPORTANT: intrinsics must be on SAME device as pose/depth inside depth_to_pointcloud
    K_t = torch.from_numpy(K.astype(np.float32)).to(device)

    mapper = SemanticVoxelMap(
        encoder=enc,
        intrinsics_3x3=K_t,
        vox_size=args.vox_size,
        vox_accum_period=1,
        max_pts_per_frame=args.max_pts_per_frame,
    )

    sam_viz_path = None
    sam_mask_generator = None
    sam_saved = 0
    if args.sam_viz_count > 0:
        sam_viz_path = os.path.join(args.out_dir, args.sam_viz_dir)
        os.makedirs(sam_viz_path, exist_ok=True)
        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_ckpt).to(device=device).eval()
        sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            points_per_batch=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            min_mask_region_area=25,
        )

    for i, (rgb_path, depth_path, pose_c2w) in enumerate(frames):
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            raise FileNotFoundError(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if rgb_hw is not None and (rgb.shape[0], rgb.shape[1]) != rgb_hw:
            H, W = rgb_hw
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

        if sam_mask_generator is not None and sam_saved < args.sam_viz_count:
            sam_masks = sam_mask_generator.generate(rgb)
            sam_overlay = render_sam_overlay(rgb, sam_masks)
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            sam_out = os.path.join(sam_viz_path, f"{sam_saved:02d}_{stem}.png")
            cv2.imwrite(sam_out, cv2.cvtColor(sam_overlay, cv2.COLOR_RGB2BGR))
            sam_saved += 1

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3,H,W)

        depth_m = read_depth_m(depth_path, depth_scale=args.depth_scale)
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).float()  # (1,H,W)

        pose_t = torch.from_numpy(pose_c2w.astype(np.float32))  # (4,4)

        # ---- Make RayFronts happy: batch dims + device ----
        rgb_t = rgb_t.unsqueeze(0).to(device)     # (1,3,H,W)
        depth_t = depth_t.unsqueeze(0).to(device) # (1,1,H,W)
        pose_t = pose_t.unsqueeze(0).to(device)   # (1,4,4)

        mapper.process_posed_rgbd(
            rgb_img=rgb_t,
            depth_img=depth_t,
            pose_4x4=pose_t,
        )

        if (i + 1) % 10 == 0:
            print(f"[INFO] processed {i+1}/{len(frames)}")

    xyz = mapper.global_vox_xyz.detach().cpu()
    feats = mapper.global_vox_feat.detach().cpu()

    out_pt = os.path.join(args.out_dir, "map.pt")
    torch.save({"feats_xyz": xyz, "feats_feats": feats, "K": K.astype(np.float32)}, out_pt)
    print(f"[OK] wrote {out_pt}")
    print(f"[OK] feats_xyz={tuple(xyz.shape)} feats_feats={tuple(feats.shape)}")
    if sam_viz_path is not None:
        print(f"[OK] SAM visualizations={sam_saved} dir={sam_viz_path}")


if __name__ == "__main__":
    main()


#  export PYTHONPATH="$(pwd)/evaluation/3d/RayFronts:$(pwd):$PYTHONPATH"


#     python tools/build_map_one_scene.py \
#   --data_root /ocean/projects/cis220039p/hguo7/datasets/SceneVerse/ScanNetPP_singleview_no_skip_frame \
#   --scene 5a269ba6fe \
#   --out_dir /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe \
#   --device cuda \
#   --frame_skip 0 \
#   --amp