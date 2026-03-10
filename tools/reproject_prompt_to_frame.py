#!/usr/bin/env python3
"""
Reproject prompt-selected map voxels back into a chosen RGB frame.

Given a saved map.pt (feats_xyz + feats_feats), this script:
1) Computes prompt similarity per voxel.
2) Keeps voxels above a threshold (default 0.065).
3) Projects those 3D points into one selected frame.
4) Saves an overlay image with highlighted projected points.

Example:
  python tools/reproject_prompt_to_frame.py \
    --map_pt /path/to/map.pt \
    --data_root /path/to/SceneVerse/ScanNetPP_singleview_no_skip_frame \
    --scene 5a269ba6fe \
    --frame_idx 0 \
    --prompt chair \
    --out_img /path/to/reproj_chair.png \
    --device cuda \
    --amp
"""

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from segment_anything import SamPredictor, sam_model_registry

from radseg.radseg import RADSegEncoder


def _sample_component_points(xs: np.ndarray, ys: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n = int(xs.shape[0])
    if n <= max_points:
        return np.stack([xs, ys], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    keep = rng.choice(n, size=max_points, replace=False)
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)


def _render_instance_overlay(rgb: np.ndarray, inst_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    ids = [i for i in np.unique(inst_map).tolist() if int(i) > 0]
    if len(ids) == 0:
        return rgb.copy()
    rng = np.random.default_rng(0)
    for inst_id in ids:
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8).astype(np.float32)
        m = inst_map == int(inst_id)
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_instance_mask_vis(inst_map: np.ndarray) -> np.ndarray:
    h, w = inst_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    ids = [i for i in np.unique(inst_map).tolist() if int(i) > 0]
    if len(ids) == 0:
        return vis
    rng = np.random.default_rng(1)
    for inst_id in ids:
        vis[inst_map == int(inst_id)] = rng.integers(32, 256, size=(3,), dtype=np.uint8)
    return vis


def _render_input_mask_overlay(rgb: np.ndarray, input_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = input_mask.astype(bool)
    if np.any(m):
        color = np.array([255.0, 255.0, 0.0], dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def run_sam_from_seed_mask(
    rgb: np.ndarray,
    seed_mask: np.ndarray,
    sam_model_type: str,
    sam_ckpt: str,
    device: str,
    sam_max_points: int,
    sam_min_seed_pixels: int,
    sam_min_mask_area: int,
    sam_box_pad: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    H, W = seed_mask.shape
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(device=torch.device(device)).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)

    cc_n, cc_map = cv2.connectedComponents(seed_mask.astype(np.uint8), connectivity=8)
    candidates: List[Tuple[np.ndarray, float, int]] = []

    for cc_id in range(1, int(cc_n)):
        ys, xs = np.where(cc_map == cc_id)
        seed_px = int(xs.shape[0])
        if seed_px < sam_min_seed_pixels:
            continue

        pts_xy = _sample_component_points(xs, ys, max_points=sam_max_points, seed=cc_id)
        point_labels = np.ones((pts_xy.shape[0],), dtype=np.int32)

        x0 = max(0, int(xs.min()) - sam_box_pad)
        y0 = max(0, int(ys.min()) - sam_box_pad)
        x1 = min(W - 1, int(xs.max()) + sam_box_pad)
        y1 = min(H - 1, int(ys.max()) + sam_box_pad)
        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        masks, scores, _ = predictor.predict(
            point_coords=pts_xy,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            continue
        best = int(np.argmax(scores))
        m = masks[best].astype(bool)
        if int(m.sum()) < sam_min_mask_area:
            continue
        candidates.append((m, float(scores[best]), seed_px))

    # Compose final non-overlapping instance map (higher-score instances win).
    inst_map = np.zeros((H, W), dtype=np.uint16)
    stats: List[Dict[str, float]] = []
    next_id = 1
    for m, score, seed_px in sorted(candidates, key=lambda x: x[1], reverse=True):
        place = m & (inst_map == 0)
        area = int(place.sum())
        if area < sam_min_mask_area:
            continue
        inst_map[place] = np.uint16(next_id)
        stats.append({"instance_id": float(next_id), "score": score, "area": float(area), "seed_px": float(seed_px)})
        next_id += 1

    return inst_map, stats


def run_sam_from_points(
    rgb: np.ndarray,
    points_xy: np.ndarray,
    point_scores: np.ndarray,
    sam_model_type: str,
    sam_ckpt: str,
    device: str,
    sam_max_points: int,
    sam_min_seed_pixels: int,
    sam_min_mask_area: int,
    sam_box_pad: int,
    sam_anchor_radius: float,
    sam_max_anchors: int,
    sam_point_pool: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    H, W = rgb.shape[:2]
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(device=torch.device(device)).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)

    if points_xy.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint16), []

    # Keep a manageable, high-confidence pool of points.
    order = np.argsort(-point_scores)
    keep = order[: min(int(sam_point_pool), int(order.shape[0]))]
    pool_xy = points_xy[keep].astype(np.float32)
    pool_sc = point_scores[keep].astype(np.float32)

    # Greedy anchor selection in point space (NMS by Euclidean radius).
    anchors: List[int] = []
    for idx in np.argsort(-pool_sc):
        if len(anchors) >= int(sam_max_anchors):
            break
        p = pool_xy[int(idx)]
        if len(anchors) == 0:
            anchors.append(int(idx))
            continue
        a = pool_xy[np.array(anchors, dtype=np.int32)]
        if np.all(np.sum((a - p[None, :]) ** 2, axis=1) > float(sam_anchor_radius) ** 2):
            anchors.append(int(idx))

    candidates: List[Tuple[np.ndarray, float, int]] = []
    for aidx in anchors:
        p = pool_xy[aidx]
        d2 = np.sum((pool_xy - p[None, :]) ** 2, axis=1)
        nbr = np.where(d2 <= float(sam_anchor_radius) ** 2)[0]
        seed_px = int(nbr.shape[0])
        if seed_px < int(sam_min_seed_pixels):
            continue

        if nbr.shape[0] > int(sam_max_points):
            nbr = nbr[np.argsort(-pool_sc[nbr])[: int(sam_max_points)]]

        pts_xy = pool_xy[nbr].astype(np.float32)
        point_labels = np.ones((pts_xy.shape[0],), dtype=np.int32)

        x0 = max(0, int(np.floor(pts_xy[:, 0].min())) - sam_box_pad)
        y0 = max(0, int(np.floor(pts_xy[:, 1].min())) - sam_box_pad)
        x1 = min(W - 1, int(np.ceil(pts_xy[:, 0].max())) + sam_box_pad)
        y1 = min(H - 1, int(np.ceil(pts_xy[:, 1].max())) + sam_box_pad)
        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        masks, scores, _ = predictor.predict(
            point_coords=pts_xy,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            continue

        best = int(np.argmax(scores))
        m = masks[best].astype(bool)
        if int(m.sum()) < int(sam_min_mask_area):
            continue
        candidates.append((m, float(scores[best]), seed_px))

    inst_map = np.zeros((H, W), dtype=np.uint16)
    stats: List[Dict[str, float]] = []
    next_id = 1
    for m, score, seed_px in sorted(candidates, key=lambda x: x[1], reverse=True):
        place = m & (inst_map == 0)
        area = int(place.sum())
        if area < int(sam_min_mask_area):
            continue
        inst_map[place] = np.uint16(next_id)
        stats.append({"instance_id": float(next_id), "score": score, "area": float(area), "seed_px": float(seed_px)})
        next_id += 1

    return inst_map, stats


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

            keys = {name, name.lower(), stem, stem.lower()}
            for k in keys:
                if k:
                    poses[k] = T
    return poses


def load_scene_frame(data_root: str, scene: str, frame_idx: int) -> Tuple[str, np.ndarray]:
    scene_root = os.path.join(data_root, scene)
    color_dir = os.path.join(scene_root, "color")
    poses_path = os.path.join(scene_root, "poses.csv")

    if not os.path.exists(color_dir):
        raise FileNotFoundError(color_dir)
    if not os.path.exists(poses_path):
        raise FileNotFoundError(poses_path)

    poses = read_poses_csv_sceneverse(poses_path, pose_is_w2c=True)
    color_files = sorted([f for f in os.listdir(color_dir) if f.lower().endswith(".png")])
    if len(color_files) == 0:
        raise RuntimeError(f"No color pngs in {color_dir}")

    paired: List[Tuple[str, np.ndarray]] = []
    for cf in color_files:
        stem = os.path.splitext(cf)[0]
        pose = poses.get(stem, poses.get(stem.lower()))
        if pose is None:
            continue
        paired.append((os.path.join(color_dir, cf), pose))

    if len(paired) == 0:
        raise RuntimeError("No RGB/pose pairs found")
    if frame_idx < 0 or frame_idx >= len(paired):
        raise IndexError(f"frame_idx={frame_idx} out of range [0, {len(paired)-1}]")

    return paired[frame_idx]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_pt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frame_idx", type=int, required=True)
    ap.add_argument("--out_img", required=True)

    ap.add_argument("--prompt", default="chair")
    ap.add_argument("--threshold", type=float, default=0.065)
    ap.add_argument(
        "--pre_filter_prompts",
        default="floor,ceiling",
        help="Comma-separated prompts used to filter points before the main prompt (e.g., 'floor,ceiling')",
    )
    ap.add_argument(
        "--pre_filter_threshold",
        type=float,
        default=0.065,
        help="Similarity threshold for pre-filter prompts; points above are removed",
    )
    ap.add_argument(
        "--occlusion_eps",
        type=float,
        default=0.03,
        help="Depth tolerance in meters for visibility test (larger keeps more near-occluded points)",
    )
    ap.add_argument("--point_size", type=int, default=1, help="Radius in pixels for highlighted points")
    ap.add_argument("--rgb_h", type=int, default=-1)
    ap.add_argument("--rgb_w", type=int, default=-1)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--sam_model_type", default="vit_h")
    ap.add_argument("--sam_ckpt", default="checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--sam_prompt_mode", choices=["points", "mask"], default="mask")
    ap.add_argument("--sam_max_points", type=int, default=32)
    ap.add_argument("--sam_min_seed_pixels", type=int, default=8)
    ap.add_argument("--sam_min_mask_area", type=int, default=64)
    ap.add_argument("--sam_box_pad", type=int, default=8)
    ap.add_argument("--sam_anchor_radius", type=float, default=24.0)
    ap.add_argument("--sam_max_anchors", type=int, default=24)
    ap.add_argument("--sam_point_pool", type=int, default=3000)
    ap.add_argument("--sam_overlay_alpha", type=float, default=0.55)
    ap.add_argument(
        "--sam_out_mask",
        default="",
        help="Output path for SAM instance-id mask PNG (uint16). Default: <out_img stem>_sam_inst.png",
    )
    ap.add_argument(
        "--sam_out_overlay",
        default="",
        help="Output path for SAM overlay PNG. Default: <out_img stem>_sam_overlay.png",
    )
    ap.add_argument(
        "--sam_out_mask_vis",
        default="",
        help="Output path for SAM mask visualization PNG. Default: <out_img stem>_sam_mask_vis.png",
    )
    ap.add_argument(
        "--sam_inmask_out",
        default="",
        help="Output path for SAM input binary mask PNG (0/255). Default: <out_img stem>_sam_input_mask.png",
    )
    ap.add_argument(
        "--sam_inmask_overlay_out",
        default="",
        help="Output path for SAM input mask overlay PNG. Default: <out_img stem>_sam_input_mask_overlay.png",
    )
    ap.add_argument("--sam_inmask_overlay_alpha", type=float, default=0.55)
    return ap.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_img) or ".", exist_ok=True)

    # Keep legacy load behavior explicit for compatibility with existing map.pt files.
    d = torch.load(args.map_pt, map_location="cpu", weights_only=False)
    feats_xyz = d["feats_xyz"]
    feats = d["feats_feats"]
    K = d["K"]

    if not isinstance(feats_xyz, torch.Tensor):
        feats_xyz = torch.tensor(feats_xyz)
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)
    if not isinstance(K, np.ndarray):
        K = np.asarray(K, dtype=np.float32)
    K = K.astype(np.float32)

    if feats_xyz.shape[0] != feats.shape[0]:
        raise RuntimeError(
            f"map.pt inconsistent: feats_xyz has {feats_xyz.shape[0]} points, "
            f"feats_feats has {feats.shape[0]} points"
        )

    rgb_path, pose_c2w = load_scene_frame(args.data_root, args.scene, args.frame_idx)
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = rgb.shape[:2]

    if args.rgb_h > 0 and args.rgb_w > 0:
        rgb = cv2.resize(rgb, (args.rgb_w, args.rgb_h), interpolation=cv2.INTER_AREA)

    H, W = rgb.shape[:2]

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

    lang_feats = enc.align_spatial_features_with_language(
        feats.unsqueeze(-1).unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)
    lang_feats = lang_feats / (lang_feats.norm(dim=-1, keepdim=True) + 1e-12)

    # Pre-filter away floor/ceiling-like points before target prompting.
    pre_filter_prompts = [p.strip() for p in str(args.pre_filter_prompts).split(",") if p.strip()]
    pre_filter_removed = 0
    keep_non_bg = torch.ones((lang_feats.shape[0],), dtype=torch.bool, device=lang_feats.device)
    if len(pre_filter_prompts) > 0:
        if hasattr(enc, "encode_labels"):
            pre_text = enc.encode_labels(pre_filter_prompts)
        else:
            pre_text = enc.encode_prompts(pre_filter_prompts)
        pre_text = pre_text / (pre_text.norm(dim=-1, keepdim=True) + 1e-12)
        # (N,D) @ (D,P) -> (N,P), then max over P
        pre_scores = lang_feats @ pre_text.transpose(0, 1)
        pre_max = pre_scores.max(dim=1).values
        keep_non_bg = pre_max < float(args.pre_filter_threshold)
        pre_filter_removed = int((~keep_non_bg).sum().item())

    if hasattr(enc, "encode_labels"):
        text = enc.encode_labels([args.prompt])
    else:
        text = enc.encode_prompts([args.prompt])
    text = text / (text.norm(dim=-1, keepdim=True) + 1e-12)

    scores = (lang_feats @ text[0].unsqueeze(-1)).squeeze(-1)

    mask = (scores >= float(args.threshold)) & keep_non_bg
    sel_n = int(mask.sum().item())
    total_n = int(scores.shape[0])
    if sel_n == 0:
        raise RuntimeError(f"No points selected at threshold={args.threshold}. Try lowering it.")

    # Index feats_xyz with a mask on the same device to avoid CPU/CUDA mismatch.
    mask_xyz = mask.to(feats_xyz.device)
    sel_xyz = feats_xyz[mask_xyz].detach().cpu().numpy().astype(np.float32)
    sel_scores = scores[mask].detach().cpu().numpy().astype(np.float32)

    smin = float(sel_scores.min())
    smax = float(sel_scores.max())
    sel_score01 = (sel_scores - smin) / (smax - smin + 1e-12)

    w2c = np.linalg.inv(pose_c2w).astype(np.float32)
    Rcw = w2c[:3, :3]
    tcw = w2c[:3, 3]
    xyz_cam = (sel_xyz @ Rcw.T) + tcw[None, :]

    z = xyz_cam[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    xyz_cam = xyz_cam[valid]
    sel_score01 = sel_score01[valid]

    if xyz_cam.shape[0] == 0:
        raise RuntimeError("All selected points are behind camera or invalid for chosen frame.")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # If RGB was resized, project with intrinsics scaled to match output image size.
    if args.rgb_h > 0 and args.rgb_w > 0:
        sx = float(W) / float(W0)
        sy = float(H) / float(H0)
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy

    # Build a per-pixel nearest depth map from all map points for occlusion filtering.
    all_xyz = feats_xyz.detach().cpu().numpy().astype(np.float32)
    all_cam = (all_xyz @ Rcw.T) + tcw[None, :]
    all_z = all_cam[:, 2]
    all_valid = np.isfinite(all_z) & (all_z > 1e-6)
    all_cam = all_cam[all_valid]
    all_z = all_z[all_valid]

    all_u = fx * (all_cam[:, 0] / all_cam[:, 2]) + cx
    all_v = fy * (all_cam[:, 1] / all_cam[:, 2]) + cy
    all_ui = np.round(all_u).astype(np.int32)
    all_vi = np.round(all_v).astype(np.int32)
    all_in = (all_ui >= 0) & (all_ui < W) & (all_vi >= 0) & (all_vi < H)
    all_ui = all_ui[all_in]
    all_vi = all_vi[all_in]
    all_z = all_z[all_in]

    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(zbuf.reshape(-1), all_vi * W + all_ui, all_z)

    u = fx * (xyz_cam[:, 0] / xyz_cam[:, 2]) + cx
    v = fy * (xyz_cam[:, 1] / xyz_cam[:, 2]) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_img = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

    ui = ui[in_img]
    vi = vi[in_img]
    z_img = xyz_cam[:, 2][in_img]
    score_img = sel_score01[in_img]
    if ui.shape[0] == 0:
        raise RuntimeError("No selected points project into image bounds for chosen frame.")

    # Keep only points that are on/near the first visible surface at each pixel.
    vis = z_img <= (zbuf[vi, ui] + float(args.occlusion_eps))
    ui = ui[vis]
    vi = vi[vis]
    score_img = score_img[vis]
    if ui.shape[0] == 0:
        raise RuntimeError(
            "All projected prompt points are occluded in this frame. "
            "Try another frame, lower threshold, or increase --occlusion_eps."
        )

    heat = np.zeros((H, W), dtype=np.float32)
    flat_idx = vi * W + ui
    np.maximum.at(heat.reshape(-1), flat_idx, score_img)
    seed_mask = heat > 0

    overlay = rgb.copy()
    ys, xs = np.where(heat > 0)
    if args.point_size <= 1:
        alpha = heat[ys, xs]
        overlay[ys, xs, 0] = np.clip((1.0 - alpha) * overlay[ys, xs, 0] + alpha * 255.0, 0, 255).astype(np.uint8)
        overlay[ys, xs, 1] = np.clip((1.0 - alpha) * overlay[ys, xs, 1], 0, 255).astype(np.uint8)
        overlay[ys, xs, 2] = np.clip((1.0 - alpha) * overlay[ys, xs, 2], 0, 255).astype(np.uint8)
    else:
        for x, y, a in zip(xs.tolist(), ys.tolist(), heat[ys, xs].tolist()):
            color = (0, 0, int(max(32, min(255, round(a * 255.0)))))
            cv2.circle(overlay, (x, y), int(args.point_size), color, thickness=-1, lineType=cv2.LINE_AA)

    out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out_img, out_bgr)

    out_stem, _ = os.path.splitext(args.out_img)
    sam_out_mask = args.sam_out_mask if args.sam_out_mask else f"{out_stem}_sam_inst.png"
    sam_out_overlay = args.sam_out_overlay if args.sam_out_overlay else f"{out_stem}_sam_overlay.png"
    sam_out_mask_vis = args.sam_out_mask_vis if args.sam_out_mask_vis else f"{out_stem}_sam_mask_vis.png"
    sam_inmask_out = args.sam_inmask_out if args.sam_inmask_out else f"{out_stem}_sam_input_mask.png"
    sam_inmask_overlay_out = (
        args.sam_inmask_overlay_out if args.sam_inmask_overlay_out else f"{out_stem}_sam_input_mask_overlay.png"
    )
    os.makedirs(os.path.dirname(sam_out_mask) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(sam_out_overlay) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(sam_out_mask_vis) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(sam_inmask_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(sam_inmask_overlay_out) or ".", exist_ok=True)

    input_mask_u8 = (seed_mask.astype(np.uint8) * 255)
    input_mask_overlay = _render_input_mask_overlay(
        rgb, seed_mask, alpha=float(args.sam_inmask_overlay_alpha)
    )
    cv2.imwrite(sam_inmask_out, input_mask_u8)
    cv2.imwrite(sam_inmask_overlay_out, cv2.cvtColor(input_mask_overlay, cv2.COLOR_RGB2BGR))

    points_xy = np.stack([ui, vi], axis=1).astype(np.float32)
    if args.sam_prompt_mode == "points":
        inst_map, inst_stats = run_sam_from_points(
            rgb=rgb,
            points_xy=points_xy,
            point_scores=score_img.astype(np.float32),
            sam_model_type=args.sam_model_type,
            sam_ckpt=args.sam_ckpt,
            device=args.device,
            sam_max_points=args.sam_max_points,
            sam_min_seed_pixels=args.sam_min_seed_pixels,
            sam_min_mask_area=args.sam_min_mask_area,
            sam_box_pad=args.sam_box_pad,
            sam_anchor_radius=args.sam_anchor_radius,
            sam_max_anchors=args.sam_max_anchors,
            sam_point_pool=args.sam_point_pool,
        )
    else:
        inst_map, inst_stats = run_sam_from_seed_mask(
            rgb=rgb,
            seed_mask=seed_mask,
            sam_model_type=args.sam_model_type,
            sam_ckpt=args.sam_ckpt,
            device=args.device,
            sam_max_points=args.sam_max_points,
            sam_min_seed_pixels=args.sam_min_seed_pixels,
            sam_min_mask_area=args.sam_min_mask_area,
            sam_box_pad=args.sam_box_pad,
        )
    sam_overlay = _render_instance_overlay(rgb, inst_map, alpha=float(args.sam_overlay_alpha))
    sam_mask_vis = _render_instance_mask_vis(inst_map)

    cv2.imwrite(sam_out_mask, inst_map)
    cv2.imwrite(sam_out_overlay, cv2.cvtColor(sam_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(sam_out_mask_vis, cv2.cvtColor(sam_mask_vis, cv2.COLOR_RGB2BGR))

    print(f"[OK] wrote {args.out_img}")
    print(f"[OK] wrote {sam_out_mask}")
    print(f"[OK] wrote {sam_out_overlay}")
    print(f"[OK] wrote {sam_out_mask_vis}")
    print(f"[OK] wrote {sam_inmask_out}")
    print(f"[OK] wrote {sam_inmask_overlay_out}")
    print(f"[INFO] prompt='{args.prompt}' threshold={args.threshold}")
    print(
        f"[INFO] pre_filter_prompts={pre_filter_prompts} "
        f"pre_filter_threshold={args.pre_filter_threshold} removed={pre_filter_removed}"
    )
    print(f"[INFO] selected={sel_n}/{total_n} projected_visible={len(ui)}")
    print(f"[INFO] sam_prompt_mode={args.sam_prompt_mode}")
    print(f"[INFO] sam_instances={len(inst_stats)}")
    print(f"[INFO] frame={args.frame_idx} rgb={rgb_path}")


if __name__ == "__main__":
    main()




# python tools/reproject_prompt_to_frame.py \
#   --map_pt /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe/map.pt \
#   --data_root /ocean/projects/cis220039p/hguo7/datasets/SceneVerse/ScanNetPP_singleview_no_skip_frame \
#   --scene 5a269ba6fe \
#   --frame_idx 0 \
#   --prompt chair \
#   --threshold 0.065 \
#   --occlusion_eps 0.02 \
#   --out_img /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe/reproj_chair_f0.png \
#   --sam_ckpt checkpoints/sam_vit_h_4b8939.pth \
#   --device cuda \
#   --amp
#   --occlusion_eps 0.02   
#   --occlusion_eps 0.05  

