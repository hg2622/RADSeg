#!/usr/bin/env python3
"""
tools/build_map_one_scene_with_scene_pointcloud.py

Build a voxel feature map (map.pt) for one SceneVerse scene and also export
a fused scene point cloud by projecting full-image RGB-D pixels to world.

Input scene structure:
  <data_root>/<scene_id>/
    color/*.png
    depth/*.png
    poses.csv
    intrinsics.txt

Outputs in <out_dir>:
  - map.pt: same keys as build_map_one_scene.py
  - scene_pointcloud.ply: fused world-space RGB point cloud
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from radseg.radseg import RADSegEncoder
from rayfronts.mapping.semantic_voxel_map import SemanticVoxelMap


def read_intrinsics_any(path: str, img_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

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
                    inferred_hw = (b, a)
                    break
            except Exception:
                pass
    if img_hw is None and inferred_hw is not None:
        img_hw = inferred_hw

    vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)]
    if len(vals) < 9:
        raise ValueError(f"Unrecognized intrinsics format in {path} (parsed {len(vals)} floats)")

    def good_k(kmat: np.ndarray) -> bool:
        if kmat.shape != (3, 3):
            return False
        if not (
            abs(float(kmat[2, 0])) < 1e-4
            and abs(float(kmat[2, 1])) < 1e-4
            and abs(float(kmat[2, 2] - 1.0)) < 1e-4
        ):
            return False
        fx, fy = float(kmat[0, 0]), float(kmat[1, 1])
        if fx <= 0 or fy <= 0:
            return False
        if img_hw is not None:
            h, w = img_hw
            cx, cy = float(kmat[0, 2]), float(kmat[1, 2])
            if not (-0.5 * w <= cx <= 1.5 * w and -0.5 * h <= cy <= 1.5 * h):
                return False
        return True

    k_last = np.array(vals[-9:], dtype=np.float32).reshape(3, 3)
    if good_k(k_last):
        return k_last

    best = None
    best_score = -1e18
    for s in range(0, len(vals) - 4 + 1):
        fx, fy, cx, cy = vals[s : s + 4]
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        if not good_k(k):
            continue
        score = min(fx, fy)
        if img_hw is not None:
            h, w = img_hw
            score -= abs(cx - 0.5 * w) + abs(cy - 0.5 * h)
        if score > best_score:
            best_score = score
            best = k

    if best is None:
        raise ValueError(f"Failed to parse a plausible intrinsics K from {path}")
    return best


def read_poses_csv_sceneverse(path: str, pose_is_w2c: bool = True) -> Dict[str, np.ndarray]:
    poses: Dict[str, np.ndarray] = {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        cols_l = [c.lower() for c in cols]

        name_col = None
        for c in cols:
            if c.lower() in ("image_name", "name", "frame", "image", "filename", "file"):
                name_col = c
                break
        if name_col is None:
            name_col = cols[0]

        def get(row: Dict[str, str], key_lower: str) -> str:
            for k in row.keys():
                if k.lower() == key_lower:
                    return row[k]
            raise KeyError(key_lower)

        for k in ["qw", "qx", "qy", "qz", "tx", "ty", "tz"]:
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

            rm = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            t = np.array([tx, ty, tz], dtype=np.float32)

            if pose_is_w2c:
                rm = rm.T
                t = (-rm @ t).astype(np.float32)

            t44 = np.eye(4, dtype=np.float32)
            t44[:3, :3] = rm
            t44[:3, 3] = t

            poses[stem] = t44
            poses[stem.lower()] = t44

    return poses


def read_depth_m(path: str, depth_scale: float) -> np.ndarray:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    if d.dtype == np.uint16:
        depth_m = d.astype(np.float32) / float(depth_scale)
    else:
        depth_m = d.astype(np.float32)
    depth_m[depth_m <= 0] = np.inf
    return depth_m


def backproject_rgbd_to_world(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    k: np.ndarray,
    t_c2w: np.ndarray,
    stride: int = 1,
    max_points: int = -1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    ys = np.arange(0, h, max(1, int(stride)), dtype=np.int32)
    xs = np.arange(0, w, max(1, int(stride)), dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    z = depth_m[yy, xx]
    valid = np.isfinite(z) & (z > 0)
    if int(valid.sum()) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    yy = yy[valid]
    xx = xx[valid]
    z = z[valid].astype(np.float32)

    if max_points is not None and int(max_points) > 0 and z.shape[0] > int(max_points):
        if rng is None:
            rng = np.random.default_rng(0)
        keep = rng.choice(z.shape[0], size=int(max_points), replace=False)
        yy = yy[keep]
        xx = xx[keep]
        z = z[keep]

    x = ((xx.astype(np.float32) - cx) / fx) * z
    y = ((yy.astype(np.float32) - cy) / fy) * z
    pts_c = np.stack([x, y, z], axis=1)

    rw = t_c2w[:3, :3].astype(np.float32)
    tw = t_c2w[:3, 3].astype(np.float32)
    pts_w = (pts_c @ rw.T) + tw[None, :]
    cols = rgb[yy, xx].astype(np.uint8)
    return pts_w.astype(np.float32), cols


def voxel_downsample_xyzrgb(xyz: np.ndarray, rgb: np.ndarray, voxel: float) -> Tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0 or voxel <= 0:
        return xyz, rgb
    q = np.floor(xyz / float(voxel)).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    idx = np.sort(idx)
    return xyz[idx], rgb[idx]


def write_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = int(xyz.shape[0])
    if n != int(rgb.shape[0]):
        raise ValueError("xyz and rgb must have the same number of points")

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

        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame_skip", type=int, default=1)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--rgb_h", type=int, default=-1)
    ap.add_argument("--rgb_w", type=int, default=-1)

    ap.add_argument("--vox_size", type=float, default=0.01)
    ap.add_argument("--max_pts_per_frame", type=int, default=-1)

    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--slide_crop", type=int, default=336)
    ap.add_argument("--slide_stride", type=int, default=112)
    ap.add_argument("--scga_scaling", type=float, default=10.0)
    ap.add_argument("--scra_scaling", type=float, default=10.0)

    ap.add_argument("--pc_stride", type=int, default=2)
    ap.add_argument("--pc_max_points_per_frame", type=int, default=200000)
    ap.add_argument("--pc_voxel", type=float, default=0.0)
    ap.add_argument("--pc_out_name", default="scene_pointcloud.ply")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(0)

    scene_root = os.path.join(args.data_root, args.scene)
    color_dir = os.path.join(scene_root, "color")
    depth_dir = os.path.join(scene_root, "depth")
    intr_path = os.path.join(scene_root, "intrinsics.txt")
    poses_path = os.path.join(scene_root, "poses.csv")
    for p in [color_dir, depth_dir, intr_path, poses_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    rgb_hw = (args.rgb_h, args.rgb_w) if args.rgb_h > 0 and args.rgb_w > 0 else None
    k = read_intrinsics_any(intr_path, img_hw=rgb_hw)
    poses = read_poses_csv_sceneverse(poses_path, pose_is_w2c=True)

    color_files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(".png")])
    stems = [os.path.splitext(f)[0] for f in color_files]
    stems = stems[:: max(1, int(args.frame_skip))]
    print(f"[INFO] scene={args.scene} frames={len(stems)}")

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

    k_t = torch.from_numpy(k.astype(np.float32)).to(device)
    mapper = SemanticVoxelMap(
        encoder=enc,
        intrinsics_3x3=k_t,
        vox_size=args.vox_size,
        vox_accum_period=1,
        max_pts_per_frame=args.max_pts_per_frame,
    )

    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []

    for i, stem in enumerate(stems):
        rgb_path = os.path.join(color_dir, stem + ".png")
        depth_path = os.path.join(depth_dir, stem + ".png")

        pose_c2w = poses.get(stem, poses.get(stem.lower()))
        if pose_c2w is None:
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_m = read_depth_m(depth_path, depth_scale=args.depth_scale)

        if rgb_hw is not None and (rgb.shape[0], rgb.shape[1]) != rgb_hw:
            h, w = rgb_hw
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).float()
        pose_t = torch.from_numpy(pose_c2w.astype(np.float32))

        mapper.process_posed_rgbd(
            rgb_img=rgb_t.unsqueeze(0).to(device),
            depth_img=depth_t.unsqueeze(0).to(device),
            pose_4x4=pose_t.unsqueeze(0).to(device),
        )

        pts_w, cols = backproject_rgbd_to_world(
            depth_m=depth_m,
            rgb=rgb,
            k=k,
            t_c2w=pose_c2w,
            stride=args.pc_stride,
            max_points=args.pc_max_points_per_frame,
            rng=rng,
        )
        if pts_w.shape[0] > 0:
            all_xyz.append(pts_w)
            all_rgb.append(cols)

        if (i + 1) % 10 == 0:
            print(f"[INFO] processed {i+1}/{len(stems)}")

    vox_xyz = mapper.global_vox_xyz.detach().cpu()
    vox_feat = mapper.global_vox_feat.detach().cpu()
    out_map = os.path.join(args.out_dir, "map.pt")
    torch.save({"feats_xyz": vox_xyz, "feats_feats": vox_feat, "K": k.astype(np.float32)}, out_map)
    print(f"[OK] wrote {out_map}")
    print(f"[OK] feats_xyz={tuple(vox_xyz.shape)} feats_feats={tuple(vox_feat.shape)}")

    if len(all_xyz) == 0:
        print("[WARN] no valid projected points, skip PLY export")
        return

    xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8)

    if float(args.pc_voxel) > 0:
        xyz, rgb = voxel_downsample_xyzrgb(xyz=xyz, rgb=rgb, voxel=float(args.pc_voxel))

    out_ply = os.path.join(args.out_dir, args.pc_out_name)
    write_ply_xyzrgb(out_ply, xyz, rgb)
    print(f"[OK] wrote {out_ply}")
    print(f"[OK] pointcloud_points={xyz.shape[0]}")


if __name__ == "__main__":
    main()


# Example:
# export PYTHONPATH="$(pwd)/evaluation/3d/RayFronts:$(pwd):$PYTHONPATH"
# python tools/build_map_one_scene_with_scene_pointcloud.py \
#   --data_root /ocean/projects/cis220039p/hguo7/datasets/SceneVerse/ScanNetPP_singleview_no_skip_frame \
#   --scene 5a269ba6fe \
#   --out_dir /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_map_and_pcd \
#   --device cuda \
#   --frame_skip 1 \
#   --vox_size 0.03 \
#   --pc_stride 2 \
#   --pc_voxel 0.01 \
#   --amp
