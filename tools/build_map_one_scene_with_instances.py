#!/usr/bin/env python3
"""
tools/build_map_one_scene_with_instances.py

Build a 3D voxel feature map for ONE ScanNet++ SceneVerse singleview scene,
and ALSO build a per-voxel instance_id layer (Design A) during mapping.

Input scene structure:
  <data_root>/<scene_id>/
    color/*.png
    depth/*.png
    poses.csv
    intrinsics.txt
    instance/*.png   (instance id image per frame; 0 = background)
    inst_to_sem/*.csv (not required for pure instance ids)

Output:
  <out_dir>/map.pt containing:
    feats_xyz:     (N,3) float32 (torch)
    feats_feats:   (N,C) float32 (torch)
    instance_id:   (N,) int32 (torch)   # 0 = unknown/background, 1..K = global instance ids
    K:             (3,3) float32 (numpy)
"""

import os
import re
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R

from radseg.radseg import RADSegEncoder
from rayfronts.mapping.semantic_voxel_map import SemanticVoxelMap


# ----------------------------
# Robust intrinsics parsing (same as your working build_map_one_scene.py)
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
                    inferred_hw = (b, a)  # assume W H -> (H,W)
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

            Rm = R.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
            t = np.array([tx, ty, tz], dtype=np.float32)

            if pose_is_w2c:
                Rm = Rm.T
                t = (-Rm @ t).astype(np.float32)

            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rm
            T[:3, 3] = t

            poses[stem] = T
            poses[stem.lower()] = T

    return poses


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


def load_sceneverse_singleview_frames(data_root: str, scene: str) -> List[str]:
    color_dir = os.path.join(data_root, scene, "color")
    files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(".png")])
    return [os.path.splitext(f)[0] for f in files]  # stems


# ----------------------------
# Backproject depth -> world points + flat pixel indices (stable; no RayFronts g3d)
# ----------------------------
def backproject_depth_to_world_with_flat_indices(
    depth_m: np.ndarray,      # (H,W) float32 meters, invalid=inf
    K: np.ndarray,            # (3,3)
    T_c2w: np.ndarray,        # (4,4)
    max_points: int = -1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pts_w: (P,3) float32
      flat_idx: (P,) int64  index into H*W flattened image, row-major
    """
    H, W = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # valid mask
    valid = np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

    if max_points is not None and max_points > 0 and ys.size > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        sel = rng.choice(ys.size, size=max_points, replace=False)
        ys = ys[sel]
        xs = xs[sel]

    z = depth_m[ys, xs].astype(np.float32)
    x = ((xs.astype(np.float32) - cx) / fx) * z
    y = ((ys.astype(np.float32) - cy) / fy) * z

    pts_c = np.stack([x, y, z], axis=1)  # (P,3)

    # transform to world
    Rw = T_c2w[:3, :3].astype(np.float32)
    tw = T_c2w[:3, 3].astype(np.float32)
    pts_w = (pts_c @ Rw.T) + tw[None, :]

    flat_idx = (ys.astype(np.int64) * np.int64(W) + xs.astype(np.int64))
    return pts_w.astype(np.float32), flat_idx


def voxel_key_from_xyz(xyz: np.ndarray, vox: float) -> np.ndarray:
    return np.floor(xyz / float(vox)).astype(np.int32)  # (N,3)


@dataclass
class GlobalInstance:
    inst_id: int
    centroid: np.ndarray
    count: int


class InstanceTracker:
    """
    Match per-frame instance blobs to global instance ids using centroid distance.
    """
    def __init__(self, assoc_radius: float):
        self.assoc_radius = float(assoc_radius)
        self.next_id = 1
        self.instances: Dict[int, GlobalInstance] = {}

    def match_or_create(self, centroid: np.ndarray) -> int:
        best_id = None
        best_d = 1e18
        for gid, inst in self.instances.items():
            d = float(np.linalg.norm(centroid - inst.centroid))
            if d < best_d:
                best_d = d
                best_id = gid

        if best_id is not None and best_d <= self.assoc_radius:
            inst = self.instances[best_id]
            new_count = inst.count + 1
            inst.centroid = (inst.centroid * inst.count + centroid) / new_count
            inst.count = new_count
            return best_id

        gid = self.next_id
        self.next_id += 1
        self.instances[gid] = GlobalInstance(inst_id=gid, centroid=centroid.copy(), count=1)
        return gid


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frame_skip", type=int, default=1)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--rgb_h", type=int, default=-1)
    ap.add_argument("--rgb_w", type=int, default=-1)

    # mapping
    ap.add_argument("--vox_size", type=float, default=0.05)
    ap.add_argument("--max_pts_per_frame", type=int, default=-1)

    # instance tracking + voxel vote
    ap.add_argument("--assoc_radius", type=float, default=0.60)
    ap.add_argument("--vote_min", type=int, default=5)
    ap.add_argument("--min_local_pts", type=int, default=80, help="min points for a local instance to be tracked")

    # encoder params
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--slide_crop", type=int, default=336)
    ap.add_argument("--slide_stride", type=int, default=112)
    ap.add_argument("--scga_scaling", type=float, default=10.0)
    ap.add_argument("--scra_scaling", type=float, default=10.0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    scene_root = os.path.join(args.data_root, args.scene)
    intr_path = os.path.join(scene_root, "intrinsics.txt")
    poses_path = os.path.join(scene_root, "poses.csv")

    rgb_hw = (args.rgb_h, args.rgb_w) if args.rgb_h > 0 and args.rgb_w > 0 else None
    K = read_intrinsics_any(intr_path, img_hw=rgb_hw)
    poses = read_poses_csv_sceneverse(poses_path, pose_is_w2c=True)

    stems = load_sceneverse_singleview_frames(args.data_root, args.scene)
    stems = stems[::max(1, args.frame_skip)]
    print(f"[INFO] scene={args.scene} frames={len(stems)}")

    # encoder + mapper (same style as your working script)
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
    K_t = torch.from_numpy(K.astype(np.float32)).to(device)

    mapper = SemanticVoxelMap(
        encoder=enc,
        intrinsics_3x3=K_t,
        vox_size=args.vox_size,
        vox_accum_period=1,
        max_pts_per_frame=args.max_pts_per_frame,
    )

    # votes: voxel_key -> {global_id: count}
    voxel_votes: Dict[Tuple[int, int, int], Dict[int, int]] = {}
    tracker = InstanceTracker(assoc_radius=args.assoc_radius)
    rng = np.random.default_rng(0)

    for fi, stem in enumerate(stems):
        rgb_path = os.path.join(scene_root, "color", stem + ".png")
        depth_path = os.path.join(scene_root, "depth", stem + ".png")
        inst_path = os.path.join(scene_root, "instance", stem + ".png")

        pose_c2w = poses.get(stem, poses.get(stem.lower()))
        if pose_c2w is None:
            continue

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if rgb_hw is not None and (rgb.shape[0], rgb.shape[1]) != rgb_hw:
            H, W = rgb_hw
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

        depth_m = read_depth_m(depth_path, depth_scale=args.depth_scale)

        # mapper update (batched + device)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).float()
        pose_t = torch.from_numpy(pose_c2w.astype(np.float32))

        rgb_b = rgb_t.unsqueeze(0).to(device)      # (1,3,H,W)
        depth_b = depth_t.unsqueeze(0).to(device)  # (1,1,H,W)
        pose_b = pose_t.unsqueeze(0).to(device)    # (1,4,4)

        mapper.process_posed_rgbd(rgb_img=rgb_b, depth_img=depth_b, pose_4x4=pose_b)

        # instance voting
        inst_img = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
        if inst_img is None:
            continue

        # backproject using CPU numpy (stable)
        pts_w, flat_idx = backproject_depth_to_world_with_flat_indices(
            depth_m=depth_m,
            K=K,
            T_c2w=pose_c2w,
            max_points=args.max_pts_per_frame,
            rng=rng,
        )
        if pts_w.shape[0] == 0:
            continue

        inst_flat = inst_img.reshape(-1)
        inst_ids = inst_flat[flat_idx].astype(np.int32)

        valid = inst_ids > 0
        pts_w = pts_w[valid]
        inst_ids = inst_ids[valid]
        if pts_w.shape[0] == 0:
            continue

        local_ids = np.unique(inst_ids)
        local_to_global: Dict[int, int] = {}
        for lid in local_ids:
            m = (inst_ids == lid)
            if int(m.sum()) < int(args.min_local_pts):
                continue
            c = pts_w[m].mean(axis=0)
            gid = tracker.match_or_create(c)
            local_to_global[int(lid)] = int(gid)

        if not local_to_global:
            continue

        keys = voxel_key_from_xyz(pts_w, args.vox_size)
        for k, lid in zip(keys, inst_ids):
            gid = local_to_global.get(int(lid), 0)
            if gid == 0:
                continue
            kt = (int(k[0]), int(k[1]), int(k[2]))
            dct = voxel_votes.get(kt)
            if dct is None:
                voxel_votes[kt] = {gid: 1}
            else:
                dct[gid] = dct.get(gid, 0) + 1

        if (fi + 1) % 10 == 0:
            print(f"[INFO] processed {fi+1}/{len(stems)} frames; global_instances={len(tracker.instances)}")

    # Assign instance_id to mapper voxels
    vox_xyz = mapper.global_vox_xyz.detach().cpu().numpy().astype(np.float32)
    vox_keys = voxel_key_from_xyz(vox_xyz, args.vox_size)

    instance_id = np.zeros((vox_xyz.shape[0],), dtype=np.int32)
    for i, k in enumerate(vox_keys):
        kt = (int(k[0]), int(k[1]), int(k[2]))
        dct = voxel_votes.get(kt)
        if not dct:
            continue
        gid, cnt = max(dct.items(), key=lambda kv: kv[1])
        if int(cnt) >= int(args.vote_min):
            instance_id[i] = int(gid)

    out_pt = os.path.join(args.out_dir, "map.pt")
    torch.save({
        "feats_xyz": mapper.global_vox_xyz.detach().cpu(),
        "feats_feats": mapper.global_vox_feat.detach().cpu(),
        "instance_id": torch.from_numpy(instance_id),
        "K": K.astype(np.float32),
    }, out_pt)

    uniq = sorted(set(instance_id.tolist()))
    num_inst = len([u for u in uniq if u != 0])
    print(f"[OK] wrote {out_pt}")
    print(f"[OK] voxels={vox_xyz.shape[0]} assigned={(instance_id>0).sum()} unique_instances={num_inst}")


if __name__ == "__main__":
    main()



# export PYTHONPATH="$(pwd)/evaluation/3d/RayFronts:$(pwd):$PYTHONPATH"

# python tools/build_map_one_scene_with_instances.py \
#   --data_root /ocean/projects/cis220039p/hguo7/datasets/SceneVerse/ScanNetPP_singleview_no_skip_frame \
#   --scene 5a269ba6fe \
#   --out_dir /ocean/projects/cis220039p/hguo7/radseg/radseg_prompt_vis/bcd2436daf_inst \
#   --device cuda \
#   --frame_skip 1 \
#   --amp \
#   --vox_size 0.05 \
#   --assoc_radius 1 \
#   --vote_min 2 \
#   --min_local_pts 80