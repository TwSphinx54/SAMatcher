import argparse
import csv
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

from scripts.valid_utils import (
    compute_epipolar_error,
    compute_pose_error,
    estimate_pose,
    pose_auc,
)


SKIP_ENTRIES = {'viz', 'viz_debug', 'metrics.npz'}
CSV_HEADER = [
    'AUC@5', 'AUC@10', 'AUC@20',
    'Acc@5', 'Acc@10', 'Acc@15', 'Acc@20',
    'mAP@5', 'mAP@10', 'mAP@20',
    'percentage_of_correct', 'matching_score',
]


def eval_pose_estimation(mkpts0, mkpts1, K0, K1, T_0to1):
    """Evaluate one matched pair with epipolar and relative-pose metrics."""
    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
    correct = epi_errs < 1e-5
    num_correct = int(np.sum(correct))
    precision = float(np.mean(correct)) if len(correct) > 0 else 0.0

    # RANSAC threshold in pixels at current image scale.
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, ransac=True, thresh=1.0)
    if ret is None:
        err_t, err_R = np.inf, np.inf
    else:
        R, t, _ = ret
        err_t, err_R = compute_pose_error(T_0to1, R, t)

    return {
        'error_t': float(err_t),
        'error_R': float(err_R),
        'num_correct': num_correct,
        'percentage_of_correct': precision,
        'epipolar_errors': epi_errs,
    }


def _load_reference_lines(ref_path):
    """Load optional reference pair file for per-pair result appending."""
    if osp.exists(ref_path):
        with open(ref_path, 'r') as f:
            return f.readlines()
    return None


def _find_ref_line_index(ref_lines, cands0, cands1):
    """Find the first line containing at least one candidate from each side."""
    if ref_lines is None:
        return None
    for i, line in enumerate(ref_lines):
        for a in cands0:
            if not a:
                continue
            for b in cands1:
                if b and (a in line) and (b in line):
                    return i
    return None


def _extract_image_candidates(match, key):
    """Extract robust filename candidates for reference-line lookup."""
    c0, c1 = [], []
    possible_keys = [
        ('path0', 'path1'), ('path_0', 'path_1'),
        ('img0', 'img1'), ('image0', 'image1'),
        ('name0', 'name1'), ('file0', 'file1'),
    ]
    for k0, k1 in possible_keys:
        if k0 in match and k1 in match:
            b0 = osp.basename(str(match[k0]))
            b1 = osp.basename(str(match[k1]))
            c0.append(b0)
            c1.append(b1)
            if not b0.lower().endswith('.jpg'):
                c0.append(b0 + '.jpg')
            if not b1.lower().endswith('.jpg'):
                c1.append(b1 + '.jpg')

    if isinstance(key, str) and '-' in key:
        parts = key.split('-')
        if len(parts) >= 2:
            p0, p1 = parts[0], parts[1]
            c0.extend([p0, p0 + '.jpg' if not p0.lower().endswith('.jpg') else p0])
            c1.extend([p1, p1 + '.jpg' if not p1.lower().endswith('.jpg') else p1])

    if not c0:
        c0 = ['']
    if not c1:
        c1 = ['']
    return list(dict.fromkeys(c0)), list(dict.fromkeys(c1))


def _write_ref_with_evals(ref_lines, appended_map, out_path):
    """Write reference file with appended per-pair evaluation fields."""
    if ref_lines is None:
        return
    with open(out_path, 'w') as f:
        for i, line in enumerate(ref_lines):
            stripped = line.rstrip('\n\r')
            if i in appended_map:
                f.write(f'{stripped} {appended_map[i]}\n')
            else:
                f.write(line)


def _to_scalar(x, default=np.nan):
    """Convert scalar-like values (including arrays/tensors) to float."""
    if x is None:
        return float(default)
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return float(default)


def _build_scene_row(aucs, acc5, acc10, acc15, acc20, map10, map20, pct_correct, matching_score):
    """Create one numeric metrics row for CSV output."""
    return [
        float(aucs[0]), float(aucs[1]), float(aucs[2]),
        float(acc5), float(acc10), float(acc15), float(acc20),
        float(acc5), float(map10), float(map20),
        float(pct_correct), float(matching_score),
    ]


def main(path, timing=False, ref_path='data/megadepth_scale_2.txt', csv_path='metrics.csv'):
    """Evaluate all scene-level infos.npz files under a result directory."""
    scenes = sorted(os.listdir(path))
    ref_lines = _load_reference_lines(ref_path)
    appended_map = {}

    save_dict = {}
    csv_rows = []
    all_overlap_times = []

    for scene in scenes:
        if scene in SKIP_ENTRIES:
            continue

        info_path = osp.join(path, scene, 'infos.npz')
        if not osp.exists(info_path):
            continue

        print(scene)
        with np.load(info_path, allow_pickle=True) as data:
            matches = [(k, data[k].item()) for k in data.files]

        if len(matches) == 0:
            continue

        scene_overlap_times = []
        pose_errors = []
        percentage_of_correct_points = []
        confs = []

        for key, match in tqdm(matches):
            K0, K1 = match['K0'], match['K1']
            T_0to1 = match['T_0to1']
            mkpts0, mkpts1 = match['mkpts0'], match['mkpts1']
            conf = _to_scalar(match.get('conf', np.nan), default=np.nan)

            overlap_time = _to_scalar(match.get('overlap_time', None), default=np.nan)
            if np.isfinite(overlap_time):
                scene_overlap_times.append(overlap_time)

            out_eval = eval_pose_estimation(mkpts0, mkpts1, K0, K1, T_0to1)
            pose_errors.append(max(out_eval['error_t'], out_eval['error_R']))
            confs.append(conf)
            percentage_of_correct_points.append(out_eval['percentage_of_correct'])

            if ref_lines is not None:
                c0, c1 = _extract_image_candidates(match, key)
                idx = _find_ref_line_index(ref_lines, c0, c1)
                if idx is not None:
                    try:
                        mean_epi = float(np.mean(out_eval.get('epipolar_errors', [np.nan])))
                    except Exception:
                        mean_epi = np.nan
                    appended_map[idx] = (
                        '{:.6f} {:.6f} {} {:.6f} {:.6f} {:.6f}'.format(
                            float(out_eval.get('error_t', np.nan)),
                            float(out_eval.get('error_R', np.nan)),
                            int(out_eval.get('num_correct', 0)),
                            float(out_eval.get('percentage_of_correct', 0.0)),
                            mean_epi,
                            conf,
                        )
                    )

        if len(pose_errors) == 0:
            continue

        aucs = [100.0 * x for x in pose_auc(pose_errors, [5, 10, 20])]
        pose_errors_np = np.asarray(pose_errors, dtype=float)
        confs_np = np.asarray(confs, dtype=float)
        pcorrect_np = np.asarray(percentage_of_correct_points, dtype=float)

        acc5 = float(np.mean(pose_errors_np < 5.0) * 100.0)
        acc10 = float(np.mean(pose_errors_np < 10.0) * 100.0)
        acc15 = float(np.mean(pose_errors_np < 15.0) * 100.0)
        acc20 = float(np.mean(pose_errors_np < 20.0) * 100.0)
        map10 = float(np.mean([acc5, acc10]))
        map20 = float(np.mean([acc5, acc10, acc15, acc20]))
        pct_correct = float(np.nanmean(pcorrect_np) * 100.0)
        matching_score = float(np.nanmean(confs_np) * 100.0)

        output = {
            'AUC@5': aucs[0],
            'AUC@10': aucs[1],
            'AUC@20': aucs[2],
            'Acc@5': acc5,
            'Acc@10': acc10,
            'Acc@15': acc15,
            'Acc@20': acc20,
            'mAP@5': acc5,
            'mAP@10': map10,
            'mAP@20': map20,
            'percentage_of_correct': pct_correct,
            'matching_score': matching_score,
        }

        if timing and len(scene_overlap_times) > 0:
            output['avg_overlap_time'] = float(np.mean(scene_overlap_times))
            all_overlap_times.extend(scene_overlap_times)
            print(f'Scene {scene} average overlap estimation time: {output["avg_overlap_time"]:.2f} ms')

        save_dict[scene] = output
        row = _build_scene_row(
            aucs, acc5, acc10, acc15, acc20, map10, map20, pct_correct, matching_score,
        )
        csv_rows.append(row)

        print(
            'AUC@5/10/20: {:.2f}/{:.2f}/{:.2f} | '
            'Acc@5/10/15/20: {:.2f}/{:.2f}/{:.2f}/{:.2f} | '
            'mAP@10/20: {:.2f}/{:.2f} | '
            'PctCorrect: {:.2f} | MatchScore: {:.2f}'.format(
                aucs[0], aucs[1], aucs[2],
                acc5, acc10, acc15, acc20,
                map10, map20,
                pct_correct, matching_score,
            )
        )

    if len(csv_rows) == 0:
        print('No valid scenes found for evaluation.')
        return

    csv_rows_np = np.asarray(csv_rows, dtype=float)
    mean_row = np.nanmean(csv_rows_np, axis=0)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)
        for row in csv_rows:
            writer.writerow([f'{v:.2f}' for v in row])
        writer.writerow([f'{v:.2f}' for v in mean_row.tolist()])

    print(f'Results saved to {csv_path}')
    np.savez(osp.join(path, 'metrics.npz'), save_dict)

    if ref_lines is not None:
        out_ref_path = ref_path.replace('.txt', '_eval.txt')
        _write_ref_with_evals(ref_lines, appended_map, out_ref_path)
        print(f'Per-pair evals appended and saved to {out_ref_path}')

    if timing and len(all_overlap_times) > 0:
        times = np.asarray(all_overlap_times, dtype=float)
        print('Overall timing statistics (ms):')
        print(f'  count={len(times)}, mean={np.mean(times):.2f}, median={np.median(times):.2f}, min={np.min(times):.2f}, max={np.max(times):.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pose evaluation for saved matching results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--path',
        type=str,
        default='outputs/eval/superpoint_aachen_loftr_samatcher',
        help='Path to scene folders containing infos.npz files',
    )
    parser.add_argument(
        '--timing',
        action='store_true',
        default=False,
        help='Report overlap timing statistics when available',
    )
    parser.add_argument(
        '--ref_path',
        type=str,
        default='data/megadepth_scale_2.txt',
        help='Optional reference pair file for per-pair eval append output',
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='metrics.csv',
        help='Path to output CSV file',
    )
    args = parser.parse_args()

    main(args.path, timing=args.timing, ref_path=args.ref_path, csv_path=args.csv_path)
