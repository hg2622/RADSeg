#!/usr/bin/env python3
"""
tools/build_map_one_scene_with_sam_instances.py

Build a 3D voxel feature map for ONE SceneVerse scene, while also assigning:
- per-voxel instance_id (global, cross-frame)
- per-voxel semantic label_id / label_name

Pipeline:
1) Build feature voxel map with RADSeg + SemanticVoxelMap (same core as build_map_one_scene.py).
2) Run SAM AutomaticMaskGenerator on each RGB frame to get local instance masks.
3) Classify each SAM mask with RADSeg language-aligned image features.
4) Backproject mask pixels to 3D, voxelize, and vote for (instance,label) on voxels.
5) Cross-frame association:
   - match local masks to global instance ids by centroid distance, but only for same label
   - if conflicting ids land in same/nearby voxels and labels match, merge ids
6) Save map.pt with feats + instance_id + label_id + label_names.
"""

import argparse
import csv
import itertools
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from radseg.radseg import RADSegEncoder
from rayfronts.mapping.semantic_voxel_map import SemanticVoxelMap


def color_from_id(gid: int) -> Tuple[int, int, int]:
    # Deterministic, high-contrast BGR color for a given instance id.
    seed = int(gid) * 1103515245 + 12345
    b = 64 + (seed & 0x7F)
    g = 64 + ((seed >> 7) & 0x7F)
    r = 64 + ((seed >> 14) & 0x7F)
    return int(b), int(g), int(r)


def draw_label_on_mask(
    vis_bgr: np.ndarray,
    mask: np.ndarray,
    text: str,
    color_bgr: Tuple[int, int, int],
    alpha: float = 0.40,
) -> None:
    if mask is None:
        return
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if int(mask.sum()) == 0:
        return

    # Blend mask color.
    vis_bgr[mask] = (
        (1.0 - float(alpha)) * vis_bgr[mask].astype(np.float32) + float(alpha) * np.array(color_bgr, dtype=np.float32)
    ).astype(np.uint8)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return
    cy = int(np.median(ys))
    cx = int(np.median(xs))
    cv2.putText(
        vis_bgr,
        text,
        (max(0, cx - 15), max(12, cy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis_bgr,
        text,
        (max(0, cx - 15), max(12, cy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_bgr,
        1,
        cv2.LINE_AA,
    )


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

        def get(row, key_lower: str) -> str:
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


def voxel_key_from_xyz(xyz: np.ndarray, vox_size: float) -> np.ndarray:
    return np.floor(xyz / float(vox_size)).astype(np.int32)


def backproject_depth_to_world_with_flat_indices(
    depth_m: np.ndarray,
    k: np.ndarray,
    t_c2w: np.ndarray,
    max_points: int = -1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference projection path matching build_map_one_scene_with_instances.py."""
    h, w = depth_m.shape
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

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
    pts_c = np.stack([x, y, z], axis=1)

    rw = t_c2w[:3, :3].astype(np.float32)
    tw = t_c2w[:3, 3].astype(np.float32)
    pts_w = (pts_c @ rw.T) + tw[None, :]

    flat_idx = (ys.astype(np.int64) * np.int64(w) + xs.astype(np.int64))
    return pts_w.astype(np.float32), flat_idx


def parse_labels(label_str: str) -> List[str]:
    labels = [x.strip() for x in str(label_str).split(",") if x.strip()]
    if len(labels) == 0:
        raise ValueError("--labels cannot be empty")
    return labels


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        p = self.parent.get(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        else:
            self.parent[x] = p
        return self.parent[x]

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        root = min(ra, rb)
        other = max(ra, rb)
        self.parent[other] = root
        return root


@dataclass
class GlobalInst:
    gid: int
    label_id: int
    centroid: np.ndarray
    count: int


@dataclass
class PairMergeEvidence:
    conflict_vox: int = 0
    support_frames: int = 0
    last_frame: int = -1


class InstanceTracker:
    def __init__(self, assoc_radius: float):
        self.assoc_radius = float(assoc_radius)
        self.next_gid = 1
        self.instances: Dict[int, GlobalInst] = {}

    def match_or_create(self, label_id: int, centroid: np.ndarray, uf: UnionFind) -> int:
        best_gid = None
        best_dist = 1e18
        for gid, inst in self.instances.items():
            rgid = uf.find(gid)
            if rgid != gid:
                continue
            if inst.label_id != label_id:
                continue
            d = float(np.linalg.norm(inst.centroid - centroid))
            if d < best_dist:
                best_dist = d
                best_gid = gid

        if best_gid is not None and best_dist <= self.assoc_radius:
            inst = self.instances[best_gid]
            new_count = inst.count + 1
            inst.centroid = (inst.centroid * inst.count + centroid) / float(new_count)
            inst.count = new_count
            return best_gid

        gid = self.next_gid
        self.next_gid += 1
        self.instances[gid] = GlobalInst(
            gid=gid,
            label_id=int(label_id),
            centroid=centroid.astype(np.float32).copy(),
            count=1,
        )
        uf.add(gid)
        return gid


def classify_sam_masks(
    enc: RADSegEncoder,
    rgb: np.ndarray,
    sam_masks: List[Dict],
    text_feats: torch.Tensor,
    device: torch.device,
    min_mask_pixels: int,
) -> List[int]:
    """Return predicted label_id for each sam mask index."""
    if len(sam_masks) == 0:
        return []

    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    feat_map = enc.encode_image_to_feat_map(rgb_t)
    lang_map = enc.align_spatial_features_with_language(feat_map)
    lang_map = lang_map / (lang_map.norm(dim=1, keepdim=True) + 1e-12)  # (1,D,h,w)

    _, d, h_f, w_f = lang_map.shape
    lang_hw_d = lang_map[0].permute(1, 2, 0).contiguous().view(-1, d)

    preds: List[int] = []
    for m in sam_masks:
        seg = m.get("segmentation", None)
        if seg is None:
            preds.append(-1)
            continue
        if int(seg.sum()) < int(min_mask_pixels):
            preds.append(-1)
            continue

        seg_small = cv2.resize(seg.astype(np.uint8), (w_f, h_f), interpolation=cv2.INTER_NEAREST).astype(bool)
        if int(seg_small.sum()) < 4:
            preds.append(-1)
            continue

        idx = np.where(seg_small.reshape(-1))[0]
        pix_feat = lang_hw_d[idx]
        pooled = pix_feat.mean(dim=0, keepdim=True)
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-12)
        scores = (pooled @ text_feats.transpose(0, 1)).squeeze(0)
        preds.append(int(torch.argmax(scores).item()))

    return preds


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

    ap.add_argument("--vox_size", type=float, default=0.03)
    ap.add_argument("--max_pts_per_frame", type=int, default=-1)
    ap.add_argument("--max_pts_per_mask", type=int, default=4000)

    ap.add_argument("--labels", default="chair,table,sofa,bed,cabinet,desk,shelf,door,window,picture")

    ap.add_argument("--assoc_radius", type=float, default=0.60)
    ap.add_argument("--merge_radius", type=float, default=0.10)
    ap.add_argument("--merge_min_support_frames", type=int, default=2)
    ap.add_argument("--merge_min_conflict_vox", type=int, default=80)
    ap.add_argument("--merge_min_size_ratio", type=float, default=0.12)
    ap.add_argument("--vote_min", type=int, default=2)
    ap.add_argument("--min_mask_pixels", type=int, default=80)
    ap.add_argument("--min_mask_points_3d", type=int, default=80)

    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--slide_crop", type=int, default=336)
    ap.add_argument("--slide_stride", type=int, default=112)
    ap.add_argument("--scga_scaling", type=float, default=10.0)
    ap.add_argument("--scra_scaling", type=float, default=10.0)

    ap.add_argument("--sam_model_type", default="vit_h")
    ap.add_argument("--sam_ckpt", default="checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--sam_points_per_side", type=int, default=16)
    ap.add_argument("--sam_points_per_batch", type=int, default=64)
    ap.add_argument("--sam_pred_iou_thresh", type=float, default=0.88)
    ap.add_argument("--sam_stability_score_thresh", type=float, default=0.95)
    ap.add_argument("--sam_crop_n_layers", type=int, default=0)
    ap.add_argument("--sam_min_mask_region_area", type=int, default=25)

    ap.add_argument("--save_track_label", default="chair")
    ap.add_argument("--save_track_vis", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    labels = parse_labels(args.labels)

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
    print(f"[INFO] scene={args.scene} frames={len(stems)} labels={len(labels)}")

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

    if hasattr(enc, "encode_labels"):
        text_feats = enc.encode_labels(labels)
    else:
        text_feats = enc.encode_prompts(labels)
    text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-12)

    track_label_id = -1
    if str(args.save_track_label).strip() != "":
        want = str(args.save_track_label).strip().lower()
        for li, ln in enumerate(labels):
            if ln.lower() == want:
                track_label_id = int(li)
                break

    track_root = os.path.join(args.out_dir, "tracked_label_masks")
    track_overlay_dir = os.path.join(track_root, "overlays")
    track_binary_dir = os.path.join(track_root, "binary_masks")
    track_idmap_dir = os.path.join(track_root, "id_maps")
    if args.save_track_vis and track_label_id >= 0:
        os.makedirs(track_overlay_dir, exist_ok=True)
        os.makedirs(track_binary_dir, exist_ok=True)
        os.makedirs(track_idmap_dir, exist_ok=True)
        print(f"[INFO] saving tracked label '{labels[track_label_id]}' masks to {track_root}")
    elif args.save_track_vis and track_label_id < 0:
        print(f"[WARN] --save_track_label '{args.save_track_label}' not found in labels; skip mask export")

    k_t = torch.from_numpy(k.astype(np.float32)).to(device)
    mapper = SemanticVoxelMap(
        encoder=enc,
        intrinsics_3x3=k_t,
        vox_size=args.vox_size,
        vox_accum_period=1,
        max_pts_per_frame=args.max_pts_per_frame,
    )

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_ckpt).to(device=device).eval()
    sam_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.sam_points_per_side,
        points_per_batch=args.sam_points_per_batch,
        pred_iou_thresh=args.sam_pred_iou_thresh,
        stability_score_thresh=args.sam_stability_score_thresh,
        crop_n_layers=args.sam_crop_n_layers,
        min_mask_region_area=args.sam_min_mask_region_area,
    )

    uf = UnionFind()
    tracker = InstanceTracker(assoc_radius=args.assoc_radius)
    rng = np.random.default_rng(0)

    # voxel_key -> {gid: vote_count}
    voxel_votes: Dict[Tuple[int, int, int], Dict[int, int]] = {}
    gid_label: Dict[int, int] = {}
    pair_evidence: Dict[Tuple[int, int], PairMergeEvidence] = {}

    merge_vox_rad = int(np.ceil(float(args.merge_radius) / float(args.vox_size)))
    neighbor_offsets = []
    for dx, dy, dz in itertools.product(range(-merge_vox_rad, merge_vox_rad + 1), repeat=3):
        if dx * dx + dy * dy + dz * dz <= merge_vox_rad * merge_vox_rad:
            neighbor_offsets.append((dx, dy, dz))

    for fi, stem in enumerate(stems):
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

        # Update feature voxel map.
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).float()
        pose_t = torch.from_numpy(pose_c2w.astype(np.float32))

        mapper.process_posed_rgbd(
            rgb_img=rgb_t.unsqueeze(0).to(device),
            depth_img=depth_t.unsqueeze(0).to(device),
            pose_4x4=pose_t.unsqueeze(0).to(device),
        )

        sam_masks = sam_gen.generate(rgb)
        if len(sam_masks) == 0:
            continue

        # Backproject once per frame using reference path; masks pick subsets by flat index.
        pts_w_all, flat_idx_all = backproject_depth_to_world_with_flat_indices(
            depth_m=depth_m,
            k=k,
            t_c2w=pose_c2w,
            max_points=args.max_pts_per_frame,
            rng=rng,
        )
        if pts_w_all.shape[0] == 0:
            continue

        flat_to_row = np.full((depth_m.shape[0] * depth_m.shape[1],), -1, dtype=np.int64)
        flat_to_row[flat_idx_all] = np.arange(flat_idx_all.shape[0], dtype=np.int64)

        pred_labels = classify_sam_masks(
            enc=enc,
            rgb=rgb,
            sam_masks=sam_masks,
            text_feats=text_feats,
            device=device,
            min_mask_pixels=args.min_mask_pixels,
        )

        tracked_masks: List[Tuple[int, np.ndarray]] = []

        for m_idx, sam_m in enumerate(sam_masks):
            label_id = pred_labels[m_idx]
            if label_id < 0:
                continue

            seg = sam_m.get("segmentation", None)
            if seg is None:
                continue

            valid = seg & np.isfinite(depth_m) & (depth_m > 0)
            if int(valid.sum()) < int(args.min_mask_points_3d):
                continue

            mask_flat = np.where(valid.reshape(-1))[0].astype(np.int64)
            rows = flat_to_row[mask_flat]
            rows = rows[rows >= 0]
            if rows.shape[0] < int(args.min_mask_points_3d):
                continue

            if int(args.max_pts_per_mask) > 0 and rows.shape[0] > int(args.max_pts_per_mask):
                keep = rng.choice(rows.shape[0], size=int(args.max_pts_per_mask), replace=False)
                rows = rows[keep]

            pts_w = pts_w_all[rows]
            if pts_w.shape[0] < int(args.min_mask_points_3d):
                continue

            centroid = pts_w.mean(axis=0)
            gid = tracker.match_or_create(label_id=label_id, centroid=centroid, uf=uf)
            gid_label[gid] = int(label_id)
            rgid_for_save = uf.find(gid)

            if args.save_track_vis and label_id == track_label_id:
                tracked_masks.append((int(rgid_for_save), seg.astype(bool)))

            keys = voxel_key_from_xyz(pts_w, args.vox_size)
            uniq_keys, counts = np.unique(keys, axis=0, return_counts=True)

            # Accumulate merge evidence for this local mask, do not union immediately.
            local_pair_conflict: Dict[Tuple[int, int], int] = {}

            for k3, cnt in zip(uniq_keys, counts):
                key_t = (int(k3[0]), int(k3[1]), int(k3[2]))
                rgid = uf.find(gid)

                # Look for nearby conflicting ids and collect evidence only.
                for dx, dy, dz in neighbor_offsets:
                    nk = (key_t[0] + dx, key_t[1] + dy, key_t[2] + dz)
                    dct_n = voxel_votes.get(nk)
                    if not dct_n:
                        continue
                    for other_gid in list(dct_n.keys()):
                        rother = uf.find(other_gid)
                        if rgid == rother:
                            continue
                        if gid_label.get(rgid, -1) == gid_label.get(rother, -2):
                            p = (min(rgid, rother), max(rgid, rother))
                            local_pair_conflict[p] = local_pair_conflict.get(p, 0) + int(cnt)

                gid_label[rgid] = int(label_id)
                dct = voxel_votes.get(key_t)
                if dct is None:
                    voxel_votes[key_t] = {rgid: int(cnt)}
                else:
                    dct[rgid] = dct.get(rgid, 0) + int(cnt)

            for p, cvox in local_pair_conflict.items():
                ev = pair_evidence.get(p)
                if ev is None:
                    ev = PairMergeEvidence()
                    pair_evidence[p] = ev
                ev.conflict_vox += int(cvox)
                if ev.last_frame != fi:
                    ev.support_frames += 1
                    ev.last_frame = fi

        if args.save_track_vis and track_label_id >= 0 and len(tracked_masks) > 0:
            vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            id_map = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint16)

            for ti, (tgid, tmask) in enumerate(tracked_masks):
                c = color_from_id(tgid)
                draw_label_on_mask(vis_bgr=vis_bgr, mask=tmask, text=f"id:{tgid}", color_bgr=c, alpha=0.42)

                id_val = int(min(max(tgid, 0), 65535))
                id_map[tmask] = np.uint16(id_val)

                bin_png = (tmask.astype(np.uint8) * 255)
                out_bin = os.path.join(track_binary_dir, f"{stem}_chair{ti:03d}_id{tgid}.png")
                cv2.imwrite(out_bin, bin_png)

            out_ov = os.path.join(track_overlay_dir, f"{stem}.png")
            out_id = os.path.join(track_idmap_dir, f"{stem}.png")
            cv2.imwrite(out_ov, vis_bgr)
            cv2.imwrite(out_id, id_map)

        if (fi + 1) % 10 == 0:
            print(
                f"[INFO] processed {fi+1}/{len(stems)} "
                f"global_instances={len(tracker.instances)} voxels_voted={len(voxel_votes)}"
            )

        # Explicitly drop per-frame temporaries to keep peak VRAM lower.
        del rgb_t, depth_t, pose_t
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    # Canonicalize vote table after unions.
    gid_total_votes: Dict[int, int] = {}
    for _k3, dct in voxel_votes.items():
        for gid, c in dct.items():
            rg = uf.find(gid)
            gid_total_votes[rg] = gid_total_votes.get(rg, 0) + int(c)

    merged_pairs = 0
    for (a, b), ev in pair_evidence.items():
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            continue

        la = gid_label.get(ra, -1)
        lb = gid_label.get(rb, -1)
        if la < 0 or lb < 0 or la != lb:
            continue

        if ev.support_frames < int(args.merge_min_support_frames):
            continue
        if ev.conflict_vox < int(args.merge_min_conflict_vox):
            continue

        ca = tracker.instances.get(ra)
        cb = tracker.instances.get(rb)
        if ca is None or cb is None:
            continue
        dist = float(np.linalg.norm(ca.centroid - cb.centroid))
        if dist > float(args.merge_radius):
            continue

        sa = float(gid_total_votes.get(ra, 0))
        sb = float(gid_total_votes.get(rb, 0))
        if sa <= 0 or sb <= 0:
            continue
        ratio = min(sa, sb) / max(sa, sb)
        if ratio < float(args.merge_min_size_ratio):
            continue

        root = uf.union(ra, rb)
        gid_label[root] = la
        merged_pairs += 1

    voxel_votes_canon: Dict[Tuple[int, int, int], Dict[int, int]] = {}
    for k3, dct in voxel_votes.items():
        out_dct: Dict[int, int] = {}
        for gid, c in dct.items():
            rg = uf.find(gid)
            out_dct[rg] = out_dct.get(rg, 0) + int(c)
        voxel_votes_canon[k3] = out_dct

    vox_xyz = mapper.global_vox_xyz.detach().cpu().numpy().astype(np.float32)
    vox_feat = mapper.global_vox_feat.detach().cpu()
    vox_keys = voxel_key_from_xyz(vox_xyz, args.vox_size)

    instance_id = np.zeros((vox_xyz.shape[0],), dtype=np.int32)
    label_id_arr = np.full((vox_xyz.shape[0],), -1, dtype=np.int32)

    for i, k3 in enumerate(vox_keys):
        key_t = (int(k3[0]), int(k3[1]), int(k3[2]))
        dct = voxel_votes_canon.get(key_t)
        if not dct:
            continue
        gid, cnt = max(dct.items(), key=lambda kv: kv[1])
        if int(cnt) < int(args.vote_min):
            continue
        rg = uf.find(gid)
        instance_id[i] = int(rg)
        label_id_arr[i] = int(gid_label.get(rg, -1))

    out_pt = os.path.join(args.out_dir, "map.pt")
    torch.save(
        {
            "feats_xyz": torch.from_numpy(vox_xyz),
            "feats_feats": vox_feat,
            "instance_id": torch.from_numpy(instance_id),
            "label_id": torch.from_numpy(label_id_arr),
            "label_names": labels,
            "K": k.astype(np.float32),
        },
        out_pt,
    )

    assigned = int((instance_id > 0).sum())
    uniq_inst = len(set([int(x) for x in instance_id.tolist() if int(x) > 0]))
    uniq_lbl = len(set([int(x) for x in label_id_arr.tolist() if int(x) >= 0]))
    print(f"[OK] wrote {out_pt}")
    print(
        f"[OK] voxels={vox_xyz.shape[0]} assigned={assigned} "
        f"unique_instances={uniq_inst} unique_labels={uniq_lbl} merged_pairs={merged_pairs}"
    )


if __name__ == "__main__":
    main()


# Example:
# export PYTHONPATH="$(pwd)/evaluation/3d/RayFronts:$(pwd):$PYTHONPATH"
# python tools/build_map_one_scene_with_sam_instances.py \
#   --data_root /ocean/projects/cis220039p/hguo7/datasets/SceneVerse/ScanNetPP_singleview_no_skip_frame \
#   --scene 5a269ba6fe \
#   --out_dir /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_sam_inst \
#   --labels "chair,table,sofa,bed,cabinet,desk,shelf,door,window,picture" \
#   --device cuda \
#   --frame_skip 1 \
#   --vox_size 0.03 \
#   --assoc_radius 0.6 \
#   --merge_radius 0.10 \
#   --vote_min 2 \
#   --amp \
#   --save_track_vis
