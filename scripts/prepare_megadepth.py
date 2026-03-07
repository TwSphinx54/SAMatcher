import os
import argparse
import random
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from utils.io import read_mask, read_cams, load_pfm
from utils.geom import grid_positions, relative_pose, warp


def unhash_int_pair(h):
    a = h // 100000
    b = h % 100000
    return a, b


def binary_downsample_to_grid(mask_full, grid_size):
    """Downsample a full-resolution binary mask to a grid mask."""
    grid = cv2.resize(mask_full.astype(np.uint8), (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    return grid > 0


def compute_depth_covis_grids(root, cidx0, cidx1, cam_dict, grid_size):
    """Compute depth-based co-visible masks for a camera pair on a fixed grid."""
    def load_depth(root_dir, idx):
        p = os.path.join(root_dir, 'depths', f'{str(idx).zfill(8)}.pfm')
        if not os.path.exists(p):
            p = os.path.join(root_dir, 'rendered_depths', f'{str(idx).zfill(8)}.pfm')
            if not os.path.exists(p):
                return None
        return load_pfm(p)

    depth0 = load_depth(root, cidx0)
    depth1 = load_depth(root, cidx1)
    if depth0 is None or depth1 is None:
        return None, None

    K0, t0, R0, _, ori_sz0 = cam_dict[cidx0]
    K1, t1, R1, _, ori_sz1 = cam_dict[cidx1]
    rel_R01, rel_t01 = relative_pose([R0, t0], [R1, t1])

    scale0_x = float(ori_sz0[0]) / float(depth0.shape[1])
    scale0_y = float(ori_sz0[1]) / float(depth0.shape[0])
    scale1_x = float(ori_sz1[0]) / float(depth1.shape[1])
    scale1_y = float(ori_sz1[1]) / float(depth1.shape[0])

    r_K0 = K0.copy()
    r_K0[0, :] = r_K0[0, :] / scale0_x
    r_K0[1, :] = r_K0[1, :] / scale0_y
    r_K1 = K1.copy()
    r_K1[0, :] = r_K1[0, :] / scale1_x
    r_K1[1, :] = r_K1[1, :] / scale1_y

    pos0_grid = grid_positions(depth0.shape[0], depth0.shape[1])
    pos0_valid, _, _ = warp(pos0_grid, np.concatenate([rel_R01, rel_t01], axis=-1), depth0, r_K0, depth1, r_K1)
    pos0_valid = np.round(pos0_valid).astype(np.int32)
    H0, W0 = depth0.shape[:2]
    v0 = (pos0_valid[:, 0] >= 0) & (pos0_valid[:, 0] < H0) & (pos0_valid[:, 1] >= 0) & (pos0_valid[:, 1] < W0)
    yy0 = np.clip(pos0_valid[v0, 0], 0, H0 - 1)
    xx0 = np.clip(pos0_valid[v0, 1], 0, W0 - 1)
    mask0_full = np.zeros((H0, W0), dtype=bool)
    mask0_full[yy0, xx0] = True

    pos1_grid = grid_positions(depth1.shape[0], depth1.shape[1])
    rel_R10, rel_t10 = relative_pose([R1, t1], [R0, t0])
    pos1_valid, _, _ = warp(pos1_grid, np.concatenate([rel_R10, rel_t10], axis=-1), depth1, r_K1, depth0, r_K0)
    pos1_valid = np.round(pos1_valid).astype(np.int32)
    H1, W1 = depth1.shape[:2]
    v1 = (pos1_valid[:, 0] >= 0) & (pos1_valid[:, 0] < H1) & (pos1_valid[:, 1] >= 0) & (pos1_valid[:, 1] < W1)
    yy1 = np.clip(pos1_valid[v1, 0], 0, H1 - 1)
    xx1 = np.clip(pos1_valid[v1, 1], 0, W1 - 1)
    mask1_full = np.zeros((H1, W1), dtype=bool)
    mask1_full[yy1, xx1] = True

    covis0_grid = binary_downsample_to_grid(mask0_full, grid_size)
    covis1_grid = binary_downsample_to_grid(mask1_full, grid_size)
    return covis0_grid, covis1_grid


def grid_bbox(mask_grid):
    ys, xs = np.where(mask_grid > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def grid_bbox_to_image_bbox(bbox_g, grid_wh, img_wh):
    """Map inclusive grid bbox to inclusive pixel bbox."""
    gx0, gy0, gx1, gy1 = bbox_g
    gw, gh = grid_wh
    W, H = img_wh
    sx = float(W) / float(gw)
    sy = float(H) / float(gh)
    x0_pix = int(np.floor(gx0 * sx))
    y0_pix = int(np.floor(gy0 * sy))
    x1_pix = int(np.ceil((gx1 + 1) * sx) - 1)
    y1_pix = int(np.ceil((gy1 + 1) * sy) - 1)
    x0_pix = max(0, min(W - 1, x0_pix))
    y0_pix = max(0, min(H - 1, y0_pix))
    x1_pix = max(0, min(W - 1, x1_pix))
    y1_pix = max(0, min(H - 1, y1_pix))
    return [int(x0_pix), int(y0_pix), int(x1_pix), int(y1_pix)]


def make_rel(path):
    rel = os.path.relpath(path, start='.')
    return rel if rel.startswith('.') else f'./{rel}'


def stage1_select_pairs(mask_dict, cam_dict, undist_img_dir, th1):
    """Group candidate pairs by mask area ratio and sample up to `th1` per group."""
    groups = defaultdict(list)
    for key, pm in mask_dict.items():
        if isinstance(key, tuple) and len(key) == 2:
            a, b = int(key[0]), int(key[1])
        elif isinstance(key, (int, np.integer)):
            a, b = unhash_int_pair(int(key))
        else:
            continue

        if a not in cam_dict or b not in cam_dict:
            continue

        basename_a = str(a).zfill(8)
        basename_b = str(b).zfill(8)
        img_a = os.path.join(undist_img_dir, basename_a + '.jpg')
        img_b = os.path.join(undist_img_dir, basename_b + '.jpg')
        if not (os.path.exists(img_a) and os.path.exists(img_b)):
            continue

        pm = np.asarray(pm).astype(bool)
        half = pm.size // 2
        gsz = int(round(np.sqrt(half)))
        if gsz * gsz != half:
            continue

        m0 = pm[:half].reshape(gsz, gsz)
        m1 = pm[half:].reshape(gsz, gsz)
        s0 = int(np.count_nonzero(m0))
        s1 = int(np.count_nonzero(m1))
        if min(s0, s1) <= 0:
            continue

        ratio = float(max(s0, s1)) / float(min(s0, s1))
        gid = 1 if ratio < 2 else (2 if ratio < 3 else 3)
        groups[gid].append((a, b, gsz))

    selected = defaultdict(list)
    for gid, pairs in groups.items():
        random.shuffle(pairs)
        selected[gid] = pairs[:th1] if th1 and th1 > 0 else pairs
    return selected


def stage2_select_pairs(scene_root, cam_dict, selected_stage1, th2=None):
    """Re-group candidates using depth-based co-visibility masks."""
    selected_stage2 = defaultdict(list)
    total_pairs = sum(len(pairs) for pairs in selected_stage1.values())
    pbar = tqdm(total=total_pairs, desc=f"[{os.path.basename(scene_root)}] Stage2 depth masks", leave=False)

    for _, pairs in selected_stage1.items():
        grouped2 = defaultdict(list)
        for a, b, gsz in pairs:
            cov0, cov1 = compute_depth_covis_grids(scene_root, a, b, cam_dict, gsz)
            pbar.update(1)
            if cov0 is None or cov1 is None:
                continue

            s0 = int(np.count_nonzero(cov0))
            s1 = int(np.count_nonzero(cov1))
            if min(s0, s1) <= 0:
                continue

            ratio = float(max(s0, s1)) / float(min(s0, s1))
            gid2 = 1 if ratio < 2 else (2 if ratio < 3 else 3)
            grouped2[gid2].append((a, b, gsz, cov0, cov1))

        for gid2 in sorted(grouped2.keys()):
            random.shuffle(grouped2[gid2])
            take = grouped2[gid2] if (th2 is None or th2 <= 0) else grouped2[gid2][:th2]
            selected_stage2[gid2].extend(take)

    pbar.close()
    return selected_stage2


def find_scenes(root):
    """Find scene folders under `root` or `root/data`."""
    if os.path.exists(os.path.join(root, 'geolabel', 'cameras.txt')):
        return [root]

    data_dir = os.path.join(root, 'data')
    cands = [os.path.join(data_dir, d) for d in os.listdir(data_dir)] if os.path.isdir(data_dir) \
        else [os.path.join(root, d) for d in os.listdir(root)]

    scenes = []
    for p in cands:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'geolabel', 'cameras.txt')):
            scenes.append(p)
    return sorted(scenes)


def _pose44(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(-1)[:3]
    return T


def _collect_candidates_for_scene(scene_root, th1_scene, seed=0):
    random.seed(seed)
    undist_img_dir = os.path.join(scene_root, 'undist_images')
    mask_path = os.path.join(scene_root, 'geolabel', 'mask.bin')
    cam_path = os.path.join(scene_root, 'geolabel', 'cameras.txt')
    if not (os.path.exists(mask_path) and os.path.exists(cam_path)):
        return defaultdict(list), None

    mask_dict = read_mask(mask_path)
    cam_dict = read_cams(cam_path)
    stage1 = stage1_select_pairs(mask_dict, cam_dict, undist_img_dir, th1_scene)
    stage2 = stage2_select_pairs(scene_root, cam_dict, stage1, th2=None)

    by_gid = defaultdict(list)
    for gid in stage2.keys():
        for a, b, gsz, cov0, cov1 in stage2[gid]:
            by_gid[gid].append((scene_root, a, b, gsz, cov0, cov1))
    return by_gid, cam_dict


def _select_global(by_gid, target_per_group, seed=0):
    rnd = random.Random(seed)
    selected = []
    for gid in (1, 2, 3):
        cand = by_gid.get(gid, [])
        rnd.shuffle(cand)
        selected.extend(cand[:max(0, target_per_group)])
    return selected


def _save_selected_pairs(entries, is_test, cam_dict_map):
    train_lines = []
    valid_lines = []
    saved_count = 0
    sample_paths = []

    pbar = tqdm(total=len(entries), desc="Saving selected pairs", leave=False)
    for scene_root, a, b, gsz, cov0, cov1 in entries:
        basename_a = str(a).zfill(8)
        basename_b = str(b).zfill(8)
        img_a = os.path.join(scene_root, 'undist_images', basename_a + '.jpg')
        img_b = os.path.join(scene_root, 'undist_images', basename_b + '.jpg')
        ia = cv2.imread(img_a)
        ib = cv2.imread(img_b)
        pbar.update(1)
        if ia is None or ib is None:
            continue

        H0, W0 = ia.shape[:2]
        H1, W1 = ib.shape[:2]
        bbox0 = grid_bbox_to_image_bbox(grid_bbox(cov0), (gsz, gsz), (W0, H0))
        bbox1 = grid_bbox_to_image_bbox(grid_bbox(cov1), (gsz, gsz), (W1, H1))

        scene_name = os.path.basename(scene_root.rstrip('/'))
        covis_dir = os.path.join('.', 'covis', scene_name)
        os.makedirs(covis_dir, exist_ok=True)
        npz_path = os.path.join(covis_dir, f'{basename_a}_{basename_b}.npz')

        np.savez_compressed(
            npz_path,
            image_path0=make_rel(img_a),
            image_path1=make_rel(img_b),
            bbox0=np.array(bbox0, dtype=np.int32),
            bbox1=np.array(bbox1, dtype=np.int32),
            mask0=cov0.astype(np.uint8),
            mask1=cov1.astype(np.uint8),
        )

        saved_count += 1
        if len(sample_paths) < 3:
            sample_paths.append(make_rel(npz_path))

        if not is_test:
            train_lines.append(make_rel(npz_path))
            continue

        cam_dict = cam_dict_map.get(scene_root, None)
        if cam_dict is None or (a not in cam_dict) or (b not in cam_dict):
            continue

        Ka, ta, Ra, _, _ = cam_dict[a]
        Kb, tb, Rb, _, _ = cam_dict[b]
        Ta = _pose44(Ra, ta)
        Tb = _pose44(Rb, tb)
        AtoB = Tb @ np.linalg.inv(Ta)

        parts = [
            make_rel(img_a), make_rel(img_b),
            ' '.join(map(str, Ka.reshape(-1).tolist())),
            ' '.join(map(str, Kb.reshape(-1).tolist())),
            ' '.join(map(str, AtoB.reshape(-1).tolist())),
            ','.join(map(str, map(int, bbox0))),
            ','.join(map(str, map(int, bbox1))),
        ]
        valid_lines.append(' '.join(parts))

    pbar.close()
    mode = 'TEST' if is_test else 'TRAIN'
    print(f"  [{mode}] Saved {saved_count} npz files under ./covis. "
          f"Example: {', '.join(sample_paths) if sample_paths else 'N/A'}")
    return train_lines, valid_lines


def main(root, seed=0, total_train_pairs=128000, total_test_pairs=3000, stage1_factor=1.5):
    os.makedirs('./outputs', exist_ok=True)
    os.makedirs('./covis', exist_ok=True)

    scenes = find_scenes(root)
    if not scenes:
        print(f'No scenes found under {root}')
        return

    rnd = random.Random(seed)
    scenes_shuf = scenes[:]
    rnd.shuffle(scenes_shuf)
    test_scenes = scenes_shuf[:min(10, len(scenes_shuf))]
    train_scenes = scenes_shuf[min(10, len(scenes_shuf)):]

    print(f"Found {len(scenes)} scenes. Test={len(test_scenes)}, Train={len(train_scenes)}.")
    if test_scenes:
        print("Test scenes (sample): " + ', '.join(os.path.basename(s) for s in test_scenes[:5]))

    import math
    train_target_pg = max(0, int(math.ceil(float(total_train_pairs) / 3.0)))
    test_target_pg = max(0, int(math.ceil(float(total_test_pairs) / 3.0)))
    denom = max(1, len(scenes))
    th1_scene_pg = max(10, int(math.ceil(stage1_factor * float(train_target_pg + test_target_pg) / float(denom))))

    print(f"[Config] Targets/group: train={train_target_pg}, test={test_target_pg}. "
          f"Stage-1 per-scene/group={th1_scene_pg} (factor={stage1_factor}).")

    print("Collecting TRAIN candidates...")
    train_by_gid = defaultdict(list)
    for s in tqdm(train_scenes, desc="TRAIN collect"):
        by_gid, _ = _collect_candidates_for_scene(s, th1_scene_pg, seed=seed)
        for gid in by_gid.keys():
            train_by_gid[gid].extend(by_gid[gid])
    print(f"  TRAIN candidates: G1={len(train_by_gid.get(1, []))}, G2={len(train_by_gid.get(2, []))}, G3={len(train_by_gid.get(3, []))}")

    print("Collecting TEST candidates...")
    test_by_gid = defaultdict(list)
    cam_dict_map = {}
    for s in tqdm(test_scenes, desc="TEST collect"):
        by_gid, cams = _collect_candidates_for_scene(s, th1_scene_pg, seed=seed)
        if cams is not None:
            cam_dict_map[s] = cams
        for gid in by_gid.keys():
            test_by_gid[gid].extend(by_gid[gid])
    print(f"  TEST candidates: G1={len(test_by_gid.get(1, []))}, G2={len(test_by_gid.get(2, []))}, G3={len(test_by_gid.get(3, []))}")

    train_selected = _select_global(train_by_gid, train_target_pg, seed=seed)
    test_selected = _select_global(test_by_gid, test_target_pg, seed=seed)
    print(f"Selected TRAIN={len(train_selected)} (target≈{3 * train_target_pg})")
    print(f"Selected TEST={len(test_selected)} (target≈{3 * test_target_pg})")

    print("Saving TRAIN npz...")
    train_lines, _ = _save_selected_pairs(train_selected, is_test=False, cam_dict_map={})
    print("Saving TEST npz and valid entries...")
    _, valid_lines = _save_selected_pairs(test_selected, is_test=True, cam_dict_map=cam_dict_map)

    with open('./outputs/train.txt', 'w') as f:
        for p in train_lines:
            f.write(f'{p}\n')

    with open('./outputs/valid.txt', 'w') as f:
        for line in valid_lines:
            f.write(f'{line}\n')

    print(f'Saved {len(train_lines)} npz entries to ./outputs/train.txt')
    print(f'Saved {len(valid_lines)} lines to ./outputs/valid.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Dataset root or a single scene path')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--total_train_pairs', type=int, default=128000, help='Total number of TRAIN pairs')
    parser.add_argument('--total_test_pairs', type=int, default=3000, help='Total number of TEST pairs')
    parser.add_argument('--stage1_factor', type=float, default=1.5, help='Stage-1 oversampling factor')
    args = parser.parse_args()

    main(
        args.root,
        seed=args.seed,
        total_train_pairs=args.total_train_pairs,
        total_test_pairs=args.total_test_pairs,
        stage1_factor=args.stage1_factor,
    )