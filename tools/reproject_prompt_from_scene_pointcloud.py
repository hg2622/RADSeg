#!/usr/bin/env python3
"""
Reproject prompt-selected voxels using a prebuilt scene point cloud.

Inputs:
  - map.pt (feats_xyz, feats_feats, K)
  - scene point cloud PLY (XYZRGB), e.g. scene_pointcloud.ply
  - SceneVerse data_root/scene to read target frame pose and GT RGB

Behavior:
  1) Use RADSeg text similarity on map.pt voxels to select prompt voxels.
  2) Mark scene point cloud points whose voxel key matches selected map voxels.
  3) Reproject from selected frame with z-buffer occlusion:
     - Base reprojection: all scene points -> RGB image.
     - Highlight reprojection: only selected points that are front-visible.
  4) Save 3 images:
     - gt_rgb.png
     - reproj_rgb.png
     - reproj_rgb_lightup.png
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


def _sample_points_xy(points_xy: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    n = int(points_xy.shape[0])
    if n <= max_points:
        return points_xy.astype(np.float32)
    rng = np.random.default_rng(seed)
    keep = rng.choice(n, size=int(max_points), replace=False)
    return points_xy[keep].astype(np.float32)


def run_sam_instances_from_light_points(
    rgb: np.ndarray,
    points_xy: np.ndarray,
    seed_mask: Optional[np.ndarray],
    enc: RADSegEncoder,
    prompt_text_feat: torch.Tensor,
    sam_model_type: str,
    sam_ckpt: str,
    device: str,
    sam_max_points: int,
    sam_anchor_radius: float,
    sam_max_anchors: int,
    sam_point_pool: int,
    sam_box_pad: int,
    sam_min_mask_area: int,
    sam_label_threshold: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    h, w = rgb.shape[:2]
    if points_xy.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint16), []

    pool = _sample_points_xy(points_xy, max_points=max(1, int(sam_point_pool)), seed=0)

    # Greedy anchor selection to encourage multiple local instances.
    anchors: List[int] = []
    for idx in range(pool.shape[0]):
        if len(anchors) >= int(sam_max_anchors):
            break
        p = pool[idx]
        if len(anchors) == 0:
            anchors.append(idx)
            continue
        a = pool[np.array(anchors, dtype=np.int32)]
        d2 = np.sum((a - p[None, :]) ** 2, axis=1)
        if np.all(d2 > float(sam_anchor_radius) ** 2):
            anchors.append(idx)

    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt).to(device=torch.device(device)).eval()
    pred = SamPredictor(sam)
    pred.set_image(rgb)

    mask_input = None
    if seed_mask is not None:
        seed_u8 = (seed_mask.astype(np.uint8) * 255)
        seed_256 = cv2.resize(seed_u8, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        # SAM expects low-res mask logits; positive values indicate foreground prior.
        mask_logits = (seed_256 * 2.0 - 1.0).astype(np.float32)
        mask_input = mask_logits[None, :, :]

    # Build language-aligned image feature map once for class filtering.
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device=torch.device(device)) / 255.0
    feat_map = enc.encode_image_to_feat_map(rgb_t)
    lang_map = enc.align_spatial_features_with_language(feat_map)
    lang_map = lang_map / (lang_map.norm(dim=1, keepdim=True) + 1e-12)
    _, d, h_f, w_f = lang_map.shape
    lang_hw_d = lang_map[0].permute(1, 2, 0).contiguous().view(-1, d)

    candidates: List[Tuple[np.ndarray, float]] = []
    for aidx in anchors:
        p = pool[aidx]
        d2 = np.sum((pool - p[None, :]) ** 2, axis=1)
        nbr = np.where(d2 <= float(sam_anchor_radius) ** 2)[0]
        if nbr.shape[0] == 0:
            continue
        if nbr.shape[0] > int(sam_max_points):
            nbr = nbr[: int(sam_max_points)]

        pts_xy = pool[nbr].astype(np.float32)
        p_labels = np.ones((pts_xy.shape[0],), dtype=np.int32)
        x0 = max(0, int(np.floor(pts_xy[:, 0].min())) - int(sam_box_pad))
        y0 = max(0, int(np.floor(pts_xy[:, 1].min())) - int(sam_box_pad))
        x1 = min(w - 1, int(np.ceil(pts_xy[:, 0].max())) + int(sam_box_pad))
        y1 = min(h - 1, int(np.ceil(pts_xy[:, 1].max())) + int(sam_box_pad))
        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        masks, scores, _ = pred.predict(
            point_coords=pts_xy,
            point_labels=p_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            continue

        best = int(np.argmax(scores))
        m = masks[best].astype(bool)
        if int(m.sum()) < int(sam_min_mask_area):
            continue

        # Filter by prompt label score on pooled mask features.
        m_small = cv2.resize(m.astype(np.uint8), (w_f, h_f), interpolation=cv2.INTER_NEAREST).astype(bool)
        if int(m_small.sum()) < 4:
            continue
        idx = np.where(m_small.reshape(-1))[0]
        pooled = lang_hw_d[idx].mean(dim=0, keepdim=True)
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-12)
        prompt_score = float((pooled @ prompt_text_feat.transpose(0, 1)).squeeze().item())
        if prompt_score < float(sam_label_threshold):
            continue

        # rank by prompt score; stronger chair-like masks win overlaps.
        candidates.append((m, prompt_score))

    inst_map = np.zeros((h, w), dtype=np.uint16)
    inst_stats: List[Dict[str, float]] = []
    next_id = 1
    for m, pscore in sorted(candidates, key=lambda x: x[1], reverse=True):
        place = m & (inst_map == 0)
        area = int(place.sum())
        if area < int(sam_min_mask_area):
            continue
        inst_map[place] = np.uint16(next_id)
        inst_stats.append({"instance_id": float(next_id), "prompt_score": float(pscore), "area": float(area)})
        next_id += 1

    return inst_map, inst_stats


def render_instance_overlay(rgb: np.ndarray, inst_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    ids = [int(i) for i in np.unique(inst_map).tolist() if int(i) > 0]
    if len(ids) == 0:
        return rgb.copy()
    rng = np.random.default_rng(1)
    for iid in ids:
        m = inst_map == int(iid)
        color = rng.integers(32, 256, size=(3,), dtype=np.uint8).astype(np.float32)
        out[m] = (1.0 - float(alpha)) * out[m] + float(alpha) * color
    return np.clip(out, 0, 255).astype(np.uint8)


def render_binary_seed_overlay(rgb: np.ndarray, seed_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = seed_mask.astype(bool)
    if int(m.sum()) == 0:
        return rgb.copy()
    color = np.array([255.0, 255.0, 0.0], dtype=np.float32)
    out[m] = (1.0 - float(alpha)) * out[m] + float(alpha) * color
    return np.clip(out, 0, 255).astype(np.uint8)


def build_2d_prompt_seed_from_rgb(
    enc: RADSegEncoder,
    rgb: np.ndarray,
    prompt_text_feat: torch.Tensor,
    device: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (score_map_01, binary_seed_mask) on full-res image."""
    h, w = rgb.shape[:2]
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device=torch.device(device)) / 255.0
    feat_map = enc.encode_image_to_feat_map(rgb_t)
    lang_map = enc.align_spatial_features_with_language(feat_map)
    lang_map = lang_map / (lang_map.norm(dim=1, keepdim=True) + 1e-12)

    # similarity on feature grid
    sim = (lang_map[0] * prompt_text_feat[0][:, None, None]).sum(dim=0)
    sim_np = sim.detach().cpu().numpy().astype(np.float32)

    sim_up = cv2.resize(sim_np, (w, h), interpolation=cv2.INTER_LINEAR)
    seed = sim_up >= float(threshold)

    # normalize to [0,1] for visualization
    smin = float(np.min(sim_up))
    smax = float(np.max(sim_up))
    score01 = (sim_up - smin) / (smax - smin + 1e-12)
    return score01.astype(np.float32), seed.astype(bool)


def voxel_key_from_xyz(xyz: np.ndarray, vox_size: float) -> np.ndarray:
    return np.floor(xyz / float(vox_size)).astype(np.int32)


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

            for k in {name, name.lower(), stem, stem.lower()}:
                if k:
                    poses[k] = t44
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
        raise RuntimeError(f"No color png files in {color_dir}")

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


def read_ascii_ply_xyzrgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 10 or lines[0].strip() != "ply":
        raise ValueError(f"Not a valid PLY: {path}")

    vertex_count = -1
    header_end = -1
    for i, ln in enumerate(lines):
        t = ln.strip()
        if t.startswith("element vertex"):
            toks = t.split()
            vertex_count = int(toks[-1])
        if t == "end_header":
            header_end = i
            break

    if vertex_count < 0 or header_end < 0:
        raise ValueError(f"Unsupported or malformed PLY header: {path}")

    data = lines[header_end + 1 : header_end + 1 + vertex_count]
    xyz = np.zeros((vertex_count, 3), dtype=np.float32)
    rgb = np.zeros((vertex_count, 3), dtype=np.uint8)

    for i, ln in enumerate(data):
        toks = ln.strip().split()
        if len(toks) < 6:
            continue
        xyz[i, 0] = float(toks[0])
        xyz[i, 1] = float(toks[1])
        xyz[i, 2] = float(toks[2])
        rgb[i, 0] = np.uint8(int(float(toks[3])))
        rgb[i, 1] = np.uint8(int(float(toks[4])))
        rgb[i, 2] = np.uint8(int(float(toks[5])))

    return xyz, rgb


def project_points_with_zbuffer(
    xyz_w: np.ndarray,
    rgb: np.ndarray,
    w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    h: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rcw = w2c[:3, :3].astype(np.float32)
    tcw = w2c[:3, 3].astype(np.float32)
    xyz_c = (xyz_w @ rcw.T) + tcw[None, :]

    z = xyz_c[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    if int(valid.sum()) == 0:
        zbuf = np.full((h, w), np.inf, dtype=np.float32)
        return np.zeros((h, w, 3), dtype=np.uint8), zbuf, np.zeros((0,), np.int32), np.zeros((0,), np.int32)

    xyz_c = xyz_c[valid]
    rgb_v = rgb[valid]

    u = fx * (xyz_c[:, 0] / xyz_c[:, 2]) + cx
    v = fy * (xyz_c[:, 1] / xyz_c[:, 2]) + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_img = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if int(in_img.sum()) == 0:
        zbuf = np.full((h, w), np.inf, dtype=np.float32)
        return np.zeros((h, w, 3), dtype=np.uint8), zbuf, np.zeros((0,), np.int32), np.zeros((0,), np.int32)

    ui = ui[in_img]
    vi = vi[in_img]
    zz = xyz_c[:, 2][in_img]
    rgb_i = rgb_v[in_img]

    flat = vi * w + ui
    order = np.argsort(zz)  # nearest first
    flat_s = flat[order]
    ui_s = ui[order]
    vi_s = vi[order]
    zz_s = zz[order]
    rgb_s = rgb_i[order]

    uniq_flat, first_idx = np.unique(flat_s, return_index=True)
    pick_ui = ui_s[first_idx]
    pick_vi = vi_s[first_idx]
    pick_zz = zz_s[first_idx]
    pick_rgb = rgb_s[first_idx]

    img = np.zeros((h, w, 3), dtype=np.uint8)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)
    img[pick_vi, pick_ui] = pick_rgb
    zbuf[pick_vi, pick_ui] = pick_zz
    return img, zbuf, pick_ui, pick_vi


def fill_black_holes_by_dilation(
    rgb: np.ndarray,
    ksize: int,
    iterations: int,
) -> np.ndarray:
    out = rgb.copy()
    hole = np.all(out == 0, axis=2)
    if int(hole.sum()) == 0:
        return out

    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)
    dil = cv2.dilate(out, kernel, iterations=max(1, int(iterations)))
    out[hole] = dil[hole]
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_pt", required=True)
    ap.add_argument("--scene_ply", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frame_idx", type=int, required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--prompt", default="chair")
    ap.add_argument("--threshold", type=float, default=0.065)
    ap.add_argument("--pre_filter_prompts", default="floor,ceiling")
    ap.add_argument("--pre_filter_threshold", type=float, default=0.065)

    ap.add_argument("--vox_size", type=float, default=0.03)
    ap.add_argument("--occlusion_eps", type=float, default=0.02)
    ap.add_argument("--rgb_h", type=int, default=-1)
    ap.add_argument("--rgb_w", type=int, default=-1)
    ap.add_argument("--light_color", type=int, nargs=3, default=[255, 64, 64])
    ap.add_argument("--light_alpha", type=float, default=0.80)
    ap.add_argument("--reproj_dilate_ksize", type=int, default=5)
    ap.add_argument("--reproj_dilate_iter", type=int, default=1)
    ap.add_argument("--sam_model_type", default="vit_h")
    ap.add_argument("--sam_ckpt", default="checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--sam_max_points", type=int, default=64)
    ap.add_argument("--sam_anchor_radius", type=float, default=24.0)
    ap.add_argument("--sam_max_anchors", type=int, default=32)
    ap.add_argument("--sam_point_pool", type=int, default=4000)
    ap.add_argument("--sam_box_pad", type=int, default=12)
    ap.add_argument("--sam_min_mask_area", type=int, default=64)
    ap.add_argument("--sam_label_threshold", type=float, default=0.065)
    ap.add_argument("--sam_overlay_alpha", type=float, default=0.55)

    ap.add_argument("--prompt2d_threshold", type=float, default=0.065)
    ap.add_argument("--prompt2d_min_pixels", type=int, default=40)
    ap.add_argument("--seed_overlay_alpha", type=float, default=0.55)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_version", default="c-radio_v3-b")
    ap.add_argument("--lang_model", default="siglip2")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")
    return ap.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_3d = os.path.join(args.out_dir, "3d")
    out_2d = os.path.join(args.out_dir, "2d")
    os.makedirs(out_3d, exist_ok=True)
    os.makedirs(out_2d, exist_ok=True)

    d = torch.load(args.map_pt, map_location="cpu", weights_only=False)
    feats_xyz = d["feats_xyz"]
    feats = d["feats_feats"]
    k = d["K"]

    if not isinstance(feats_xyz, torch.Tensor):
        feats_xyz = torch.tensor(feats_xyz)
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)
    if not isinstance(k, np.ndarray):
        k = np.asarray(k, dtype=np.float32)
    k = k.astype(np.float32)

    if int(feats_xyz.shape[0]) != int(feats.shape[0]):
        raise RuntimeError(
            f"map.pt inconsistent: feats_xyz={feats_xyz.shape[0]} feats_feats={feats.shape[0]}"
        )

    rgb_path, pose_c2w = load_scene_frame(args.data_root, args.scene, args.frame_idx)
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb_gt = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb_gt.shape[:2]

    if args.rgb_h > 0 and args.rgb_w > 0:
        rgb_gt = cv2.resize(rgb_gt, (args.rgb_w, args.rgb_h), interpolation=cv2.INTER_AREA)
    h, w = rgb_gt.shape[:2]

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if args.rgb_h > 0 and args.rgb_w > 0:
        sx = float(w) / float(w0)
        sy = float(h) / float(h0)
        fx *= sx
        cx *= sx
        fy *= sy
        cy *= sy

    enc = RADSegEncoder(
        model_version=args.model_version,
        lang_model=args.lang_model,
        device=args.device,
        amp=args.amp,
        compile=args.compile,
        predict=False,
        sam_refinement=False,
    )

    feats_dev = feats.to(args.device)
    lang_feats = enc.align_spatial_features_with_language(
        feats_dev.unsqueeze(-1).unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)
    lang_feats = lang_feats / (lang_feats.norm(dim=-1, keepdim=True) + 1e-12)

    pre_filter_prompts = [p.strip() for p in str(args.pre_filter_prompts).split(",") if p.strip()]
    keep_non_bg = torch.ones((lang_feats.shape[0],), dtype=torch.bool, device=lang_feats.device)
    if len(pre_filter_prompts) > 0:
        if hasattr(enc, "encode_labels"):
            pre_t = enc.encode_labels(pre_filter_prompts)
        else:
            pre_t = enc.encode_prompts(pre_filter_prompts)
        pre_t = pre_t / (pre_t.norm(dim=-1, keepdim=True) + 1e-12)
        pre_scores = lang_feats @ pre_t.transpose(0, 1)
        keep_non_bg = pre_scores.max(dim=1).values < float(args.pre_filter_threshold)

    if hasattr(enc, "encode_labels"):
        txt = enc.encode_labels([args.prompt])
    else:
        txt = enc.encode_prompts([args.prompt])
    txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
    scores = (lang_feats @ txt[0].unsqueeze(-1)).squeeze(-1)

    map_sel = (scores >= float(args.threshold)) & keep_non_bg
    map_sel_n = int(map_sel.sum().item())
    if map_sel_n == 0:
        raise RuntimeError(f"No map voxels selected for prompt='{args.prompt}' at threshold={args.threshold}")

    map_xyz_sel = feats_xyz[map_sel.to(feats_xyz.device)].detach().cpu().numpy().astype(np.float32)
    map_keys = voxel_key_from_xyz(map_xyz_sel, args.vox_size)
    key_set = set((int(k0), int(k1), int(k2)) for k0, k1, k2 in map_keys.tolist())

    scene_xyz, scene_rgb = read_ascii_ply_xyzrgb(args.scene_ply)
    if scene_xyz.shape[0] == 0:
        raise RuntimeError("Scene point cloud has no points")

    scene_keys = voxel_key_from_xyz(scene_xyz, args.vox_size)
    is_light = np.zeros((scene_xyz.shape[0],), dtype=bool)
    for i, k3 in enumerate(scene_keys):
        if (int(k3[0]), int(k3[1]), int(k3[2])) in key_set:
            is_light[i] = True

    light_n = int(is_light.sum())
    if light_n == 0:
        print("[WARN] no scene points fall into selected map voxels; light-up image will match base reprojection")

    w2c = np.linalg.inv(pose_c2w).astype(np.float32)

    reproj_rgb, zbuf, _, _ = project_points_with_zbuffer(
        xyz_w=scene_xyz,
        rgb=scene_rgb,
        w2c=w2c,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        h=h,
        w=w,
    )
    reproj_rgb_dilated = fill_black_holes_by_dilation(
        rgb=reproj_rgb,
        ksize=int(args.reproj_dilate_ksize),
        iterations=int(args.reproj_dilate_iter),
    )

    # Occlusion-aware highlight: only selected points that are first-hit (within eps).
    rcw = w2c[:3, :3].astype(np.float32)
    tcw = w2c[:3, 3].astype(np.float32)
    light_xyz_c = (scene_xyz[is_light] @ rcw.T) + tcw[None, :]

    z = light_xyz_c[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    light_xyz_c = light_xyz_c[valid]

    u = fx * (light_xyz_c[:, 0] / light_xyz_c[:, 2]) + cx
    v = fy * (light_xyz_c[:, 1] / light_xyz_c[:, 2]) + cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    zi = light_xyz_c[:, 2]

    in_img = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui = ui[in_img]
    vi = vi[in_img]
    zi = zi[in_img]

    vis = zi <= (zbuf[vi, ui] + float(args.occlusion_eps))
    ui = ui[vis]
    vi = vi[vis]

    light_img = reproj_rgb.copy()
    if ui.shape[0] > 0:
        lc = np.array(args.light_color, dtype=np.float32).reshape(1, 3)
        a = float(np.clip(args.light_alpha, 0.0, 1.0))
        pix = light_img[vi, ui].astype(np.float32)
        light_img[vi, ui] = np.clip((1.0 - a) * pix + a * lc, 0.0, 255.0).astype(np.uint8)

    # -------------------------
    # 3D method (always run)
    # -------------------------
    seed3d_points_xy = (
        np.stack([ui, vi], axis=1).astype(np.float32)
        if ui.shape[0] > 0
        else np.zeros((0, 2), dtype=np.float32)
    )

    sam3d_sparse_inst, sam3d_sparse_stats = run_sam_instances_from_light_points(
        rgb=reproj_rgb,
        points_xy=seed3d_points_xy,
        seed_mask=None,
        enc=enc,
        prompt_text_feat=txt,
        sam_model_type=args.sam_model_type,
        sam_ckpt=args.sam_ckpt,
        device=args.device,
        sam_max_points=args.sam_max_points,
        sam_anchor_radius=args.sam_anchor_radius,
        sam_max_anchors=args.sam_max_anchors,
        sam_point_pool=args.sam_point_pool,
        sam_box_pad=args.sam_box_pad,
        sam_min_mask_area=args.sam_min_mask_area,
        sam_label_threshold=args.sam_label_threshold,
    )
    sam3d_dilated_inst, sam3d_dilated_stats = run_sam_instances_from_light_points(
        rgb=reproj_rgb_dilated,
        points_xy=seed3d_points_xy,
        seed_mask=None,
        enc=enc,
        prompt_text_feat=txt,
        sam_model_type=args.sam_model_type,
        sam_ckpt=args.sam_ckpt,
        device=args.device,
        sam_max_points=args.sam_max_points,
        sam_anchor_radius=args.sam_anchor_radius,
        sam_max_anchors=args.sam_max_anchors,
        sam_point_pool=args.sam_point_pool,
        sam_box_pad=args.sam_box_pad,
        sam_min_mask_area=args.sam_min_mask_area,
        sam_label_threshold=args.sam_label_threshold,
    )
    sam3d_gt_inst, sam3d_gt_stats = run_sam_instances_from_light_points(
        rgb=rgb_gt,
        points_xy=seed3d_points_xy,
        seed_mask=None,
        enc=enc,
        prompt_text_feat=txt,
        sam_model_type=args.sam_model_type,
        sam_ckpt=args.sam_ckpt,
        device=args.device,
        sam_max_points=args.sam_max_points,
        sam_anchor_radius=args.sam_anchor_radius,
        sam_max_anchors=args.sam_max_anchors,
        sam_point_pool=args.sam_point_pool,
        sam_box_pad=args.sam_box_pad,
        sam_min_mask_area=args.sam_min_mask_area,
        sam_label_threshold=args.sam_label_threshold,
    )

    sam3d_sparse_overlay = render_instance_overlay(reproj_rgb, sam3d_sparse_inst, alpha=float(args.sam_overlay_alpha))
    sam3d_dilated_overlay = render_instance_overlay(
        reproj_rgb_dilated, sam3d_dilated_inst, alpha=float(args.sam_overlay_alpha)
    )
    sam3d_gt_overlay = render_instance_overlay(rgb_gt, sam3d_gt_inst, alpha=float(args.sam_overlay_alpha))

    # -------------------------
    # 2D method (always run)
    # -------------------------
    prompt2d_score01, prompt2d_seed = build_2d_prompt_seed_from_rgb(
        enc=enc,
        rgb=reproj_rgb,
        prompt_text_feat=txt,
        device=args.device,
        threshold=float(args.prompt2d_threshold),
    )
    prompt2d_gt_score01, prompt2d_gt_seed = build_2d_prompt_seed_from_rgb(
        enc=enc,
        rgb=rgb_gt,
        prompt_text_feat=txt,
        device=args.device,
        threshold=float(args.prompt2d_threshold),
    )
    seed2d_overlay = render_binary_seed_overlay(reproj_rgb, prompt2d_seed, alpha=float(args.seed_overlay_alpha))
    seed2d_gt_overlay = render_binary_seed_overlay(rgb_gt, prompt2d_gt_seed, alpha=float(args.seed_overlay_alpha))

    if int(prompt2d_seed.sum()) < int(args.prompt2d_min_pixels):
        print(
            f"[WARN] 2D prompt seed too small ({int(prompt2d_seed.sum())} px); "
            "2D SAM outputs may be empty"
        )
    ys2, xs2 = np.where(prompt2d_seed)
    seed2d_points_xy = np.stack([xs2, ys2], axis=1).astype(np.float32) if ys2.size > 0 else np.zeros((0, 2), dtype=np.float32)

    sam2d_sparse_inst, sam2d_sparse_stats = run_sam_instances_from_light_points(
        rgb=reproj_rgb,
        points_xy=seed2d_points_xy,
        seed_mask=prompt2d_seed,
        enc=enc,
        prompt_text_feat=txt,
        sam_model_type=args.sam_model_type,
        sam_ckpt=args.sam_ckpt,
        device=args.device,
        sam_max_points=args.sam_max_points,
        sam_anchor_radius=args.sam_anchor_radius,
        sam_max_anchors=args.sam_max_anchors,
        sam_point_pool=args.sam_point_pool,
        sam_box_pad=args.sam_box_pad,
        sam_min_mask_area=args.sam_min_mask_area,
        sam_label_threshold=args.sam_label_threshold,
    )
    sam2d_dilated_inst, sam2d_dilated_stats = run_sam_instances_from_light_points(
        rgb=reproj_rgb_dilated,
        points_xy=seed2d_points_xy,
        seed_mask=prompt2d_seed,
        enc=enc,
        prompt_text_feat=txt,
        sam_model_type=args.sam_model_type,
        sam_ckpt=args.sam_ckpt,
        device=args.device,
        sam_max_points=args.sam_max_points,
        sam_anchor_radius=args.sam_anchor_radius,
        sam_max_anchors=args.sam_max_anchors,
        sam_point_pool=args.sam_point_pool,
        sam_box_pad=args.sam_box_pad,
        sam_min_mask_area=args.sam_min_mask_area,
        sam_label_threshold=args.sam_label_threshold,
    )
    ys2g, xs2g = np.where(prompt2d_gt_seed)
    seed2d_gt_points_xy = (
        np.stack([xs2g, ys2g], axis=1).astype(np.float32)
        if ys2g.size > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    sam2d_gt_inst, sam2d_gt_stats = run_sam_instances_from_light_points(
        rgb=rgb_gt,
        points_xy=seed2d_gt_points_xy,
        seed_mask=prompt2d_gt_seed,
        enc=enc,
        prompt_text_feat=txt,
        sam_model_type=args.sam_model_type,
        sam_ckpt=args.sam_ckpt,
        device=args.device,
        sam_max_points=args.sam_max_points,
        sam_anchor_radius=args.sam_anchor_radius,
        sam_max_anchors=args.sam_max_anchors,
        sam_point_pool=args.sam_point_pool,
        sam_box_pad=args.sam_box_pad,
        sam_min_mask_area=args.sam_min_mask_area,
        sam_label_threshold=args.sam_label_threshold,
    )
    sam2d_sparse_overlay = render_instance_overlay(reproj_rgb, sam2d_sparse_inst, alpha=float(args.sam_overlay_alpha))
    sam2d_dilated_overlay = render_instance_overlay(
        reproj_rgb_dilated, sam2d_dilated_inst, alpha=float(args.sam_overlay_alpha)
    )
    sam2d_gt_overlay = render_instance_overlay(rgb_gt, sam2d_gt_inst, alpha=float(args.sam_overlay_alpha))

    out_gt = os.path.join(args.out_dir, "gt_rgb.png")
    out_reproj = os.path.join(args.out_dir, "reproj_rgb.png")
    out_reproj_dilated = os.path.join(args.out_dir, "reproj_rgb_dilated.png")

    out_3d_light = os.path.join(out_3d, "reproj_rgb_lightup.png")
    out_3d_sam_sparse = os.path.join(out_3d, "sam_on_sparse.png")
    out_3d_sam_dilated = os.path.join(out_3d, "sam_on_dilated.png")
    out_3d_sam_gt = os.path.join(out_3d, "sam_on_gt.png")
    out_3d_sam_sparse_inst = os.path.join(out_3d, "sam_on_sparse_inst.png")
    out_3d_sam_dilated_inst = os.path.join(out_3d, "sam_on_dilated_inst.png")
    out_3d_sam_gt_inst = os.path.join(out_3d, "sam_on_gt_inst.png")

    out_2d_seed_overlay = os.path.join(out_2d, "seed_overlay_on_sparse.png")
    out_2d_seed_overlay_gt = os.path.join(out_2d, "seed_overlay_on_gt.png")
    out_2d_sam_sparse = os.path.join(out_2d, "sam_on_sparse.png")
    out_2d_sam_dilated = os.path.join(out_2d, "sam_on_dilated.png")
    out_2d_sam_gt = os.path.join(out_2d, "sam_on_gt.png")
    out_2d_sam_sparse_inst = os.path.join(out_2d, "sam_on_sparse_inst.png")
    out_2d_sam_dilated_inst = os.path.join(out_2d, "sam_on_dilated_inst.png")
    out_2d_sam_gt_inst = os.path.join(out_2d, "sam_on_gt_inst.png")
    out_2d_prompt_heat = os.path.join(out_2d, "seed_heat.png")
    out_2d_prompt_seed = os.path.join(out_2d, "seed_mask.png")
    out_2d_prompt_heat_gt = os.path.join(out_2d, "seed_heat_gt.png")
    out_2d_prompt_seed_gt = os.path.join(out_2d, "seed_mask_gt.png")

    cv2.imwrite(out_gt, cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_reproj, cv2.cvtColor(reproj_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_reproj_dilated, cv2.cvtColor(reproj_rgb_dilated, cv2.COLOR_RGB2BGR))

    cv2.imwrite(out_3d_light, cv2.cvtColor(light_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_3d_sam_sparse, cv2.cvtColor(sam3d_sparse_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_3d_sam_dilated, cv2.cvtColor(sam3d_dilated_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_3d_sam_gt, cv2.cvtColor(sam3d_gt_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_3d_sam_sparse_inst, sam3d_sparse_inst)
    cv2.imwrite(out_3d_sam_dilated_inst, sam3d_dilated_inst)
    cv2.imwrite(out_3d_sam_gt_inst, sam3d_gt_inst)

    cv2.imwrite(out_2d_seed_overlay, cv2.cvtColor(seed2d_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_2d_seed_overlay_gt, cv2.cvtColor(seed2d_gt_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_2d_sam_sparse, cv2.cvtColor(sam2d_sparse_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_2d_sam_dilated, cv2.cvtColor(sam2d_dilated_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_2d_sam_gt, cv2.cvtColor(sam2d_gt_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_2d_sam_sparse_inst, sam2d_sparse_inst)
    cv2.imwrite(out_2d_sam_dilated_inst, sam2d_dilated_inst)
    cv2.imwrite(out_2d_sam_gt_inst, sam2d_gt_inst)

    heat_u8 = np.clip(prompt2d_score01 * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    seed_u8 = (prompt2d_seed.astype(np.uint8) * 255)
    heat_gt_u8 = np.clip(prompt2d_gt_score01 * 255.0, 0, 255).astype(np.uint8)
    heat_gt_color = cv2.applyColorMap(heat_gt_u8, cv2.COLORMAP_JET)
    seed_gt_u8 = (prompt2d_gt_seed.astype(np.uint8) * 255)
    cv2.imwrite(out_2d_prompt_heat, heat_color)
    cv2.imwrite(out_2d_prompt_seed, seed_u8)
    cv2.imwrite(out_2d_prompt_heat_gt, heat_gt_color)
    cv2.imwrite(out_2d_prompt_seed_gt, seed_gt_u8)

    print(f"[OK] wrote {out_gt}")
    print(f"[OK] wrote {out_reproj}")
    print(f"[OK] wrote {out_reproj_dilated}")
    print(f"[OK] wrote {out_3d_light}")
    print(f"[OK] wrote {out_3d_sam_sparse}")
    print(f"[OK] wrote {out_3d_sam_dilated}")
    print(f"[OK] wrote {out_3d_sam_gt}")
    print(f"[OK] wrote {out_3d_sam_sparse_inst}")
    print(f"[OK] wrote {out_3d_sam_dilated_inst}")
    print(f"[OK] wrote {out_3d_sam_gt_inst}")
    print(f"[OK] wrote {out_2d_seed_overlay}")
    print(f"[OK] wrote {out_2d_seed_overlay_gt}")
    print(f"[OK] wrote {out_2d_sam_sparse}")
    print(f"[OK] wrote {out_2d_sam_dilated}")
    print(f"[OK] wrote {out_2d_sam_gt}")
    print(f"[OK] wrote {out_2d_sam_sparse_inst}")
    print(f"[OK] wrote {out_2d_sam_dilated_inst}")
    print(f"[OK] wrote {out_2d_sam_gt_inst}")
    print(f"[OK] wrote {out_2d_prompt_heat}")
    print(f"[OK] wrote {out_2d_prompt_seed}")
    print(f"[OK] wrote {out_2d_prompt_heat_gt}")
    print(f"[OK] wrote {out_2d_prompt_seed_gt}")
    print(f"[INFO] frame={args.frame_idx} rgb={rgb_path}")
    print(f"[INFO] prompt='{args.prompt}' threshold={args.threshold} map_selected_voxels={map_sel_n}")
    print(f"[INFO] scene_points={scene_xyz.shape[0]} light_points_voxel_match={light_n} visible_light_pixels={ui.shape[0]}")
    print(
        f"[INFO] prompt2d_threshold={args.prompt2d_threshold} "
        f"prompt2d_seed_pixels={int(prompt2d_seed.sum())}"
    )
    print(
        f"[INFO] 3d_sam_instances_sparse={len(sam3d_sparse_stats)} "
        f"3d_sam_instances_dilated={len(sam3d_dilated_stats)} "
        f"3d_sam_instances_gt={len(sam3d_gt_stats)}"
    )
    print(
        f"[INFO] 2d_sam_instances_sparse={len(sam2d_sparse_stats)} "
        f"2d_sam_instances_dilated={len(sam2d_dilated_stats)} "
        f"2d_sam_instances_gt={len(sam2d_gt_stats)}"
    )


if __name__ == "__main__":
    main()


# Example:
# python tools/reproject_prompt_from_scene_pointcloud.py \
#   --map_pt /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_map_and_pcd/map.pt \
#   --scene_ply /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_map_and_pcd/scene_pointcloud.ply \
#   --data_root /ocean/projects/cis220039p/hguo7/datasets/SceneVerse/ScanNetPP_singleview_no_skip_frame \
#   --scene 5a269ba6fe \
#   --frame_idx 0 \
#   --out_dir /ocean/projects/cis220039p/hguo7/RADSeg/radseg_prompt_vis/5a269ba6fe_map_and_pcd/reproj_chair_f0 \
#   --prompt chair \
#   --threshold 0.065 \
#   --vox_size 0.03 \
#   --occlusion_eps 0.02 \
#   --device cuda \
#   --prompt2d_threshold 0.1 \
#   --amp
