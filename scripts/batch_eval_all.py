import os
import csv
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm

# Reuse evaluation helpers from the single-pair script.
from scripts.eval_pose_estimation import eval_pose_estimation as eval_pair
from scripts.eval_pose_estimation import _load_reference_lines, _find_ref_line_index, _extract_image_candidates, _write_ref_with_evals
from scripts.valid_utils import pose_auc


def _iter_method_scenes(method_dir):
    """Yield (scene_name, infos_path) for valid scene folders under one method directory."""
    for entry in sorted(os.listdir(method_dir)):
        scene_dir = osp.join(method_dir, entry)
        if not osp.isdir(scene_dir):
            continue
        info_path = osp.join(scene_dir, 'infos.npz')
        if osp.exists(info_path):
            yield entry, info_path


def _is_method_dir(path):
    """Return True if `path` contains at least one valid scene with infos.npz."""
    if not osp.isdir(path):
        return False
    for _ in _iter_method_scenes(path):
        return True
    return False


def _collect_methods(input_dir):
    """
    Resolve methods from input:
    - if `input_dir` is already a method folder, return [input_dir]
    - otherwise return first-level subfolders that are valid methods
    """
    if _is_method_dir(input_dir):
        return [input_dir]

    methods = []
    for name in sorted(os.listdir(input_dir)):
        p = osp.join(input_dir, name)
        if osp.isdir(p) and _is_method_dir(p):
            methods.append(p)
    return methods


def _mean_epipolar_error(out_eval):
    """Compute mean epipolar error; return NaN when unavailable."""
    epi_errors = out_eval.get('epipolar_errors')
    if epi_errors is None or np.size(epi_errors) == 0:
        return np.nan
    return float(np.mean(epi_errors))


def _summary_lines(summary):
    """Format summary text block shared by all output modes."""
    return [
        "\n# Summary (cross-scene averages)\n",
        f"# num_pairs: {summary['num_pairs']}\n",
        f"# AUC@5: {summary['AUC@5']:.2f}, AUC@10: {summary['AUC@10']:.2f}, AUC@20: {summary['AUC@20']:.2f}\n",
        f"# Acc@5: {summary['Acc@5']:.2f}, Acc@10: {summary['Acc@10']:.2f}, Acc@15: {summary['Acc@15']:.2f}, Acc@20: {summary['Acc@20']:.2f}\n",
        f"# mAP@5: {summary['mAP@5']:.2f}, mAP@10: {summary['mAP@10']:.2f}, mAP@20: {summary['mAP@20']:.2f}\n",
        f"# percentage_of_correct: {summary['percentage_of_correct']:.2f}\n",
        f"# matching_score: {summary['matching_score']:.2f}\n",
    ]


def _evaluate_method(method_dir, ref_lines=None):
    pose_errors = []
    percentage_of_correct_points = []
    confs = []
    pair_lines = []
    appended_map = {}
    total_pairs = 0

    for scene, info_path in _iter_method_scenes(method_dir):
        # Auto-close npz file after each scene to keep file handles bounded.
        with np.load(info_path, allow_pickle=True) as data:
            matches = [(k, data[k].item()) for k in data.files]

        for key, match in tqdm(matches, desc=f"{osp.basename(method_dir)}:{scene}", leave=False):
            K0 = match['K0']
            K1 = match['K1']
            T_0to1 = match['T_0to1']
            mkpts0 = match['mkpts0']
            mkpts1 = match['mkpts1']
            conf = match.get('conf', np.nan)

            out_eval = eval_pair(mkpts0, mkpts1, K0, K1, T_0to1)
            pose_error = float(np.maximum(out_eval['error_t'], out_eval['error_R']))
            pose_errors.append(pose_error)
            confs.append(conf if conf is not None else np.nan)
            percentage_of_correct_points.append(float(out_eval['percentage_of_correct']))
            mean_epi = _mean_epipolar_error(out_eval)

            appended_str = "{:.6f} {:.6f} {} {:.6f} {:.6f} {:.6f}".format(
                float(out_eval.get('error_t', np.nan)),
                float(out_eval.get('error_R', np.nan)),
                int(out_eval.get('num_correct', 0)),
                float(out_eval.get('percentage_of_correct', 0.0)),
                mean_epi,
                float(conf) if conf is not None else np.nan
            )

            pair_id = f"{scene}-{str(key)}"
            pair_lines.append(f"{pair_id} {appended_str}")

            if ref_lines is not None:
                c0, c1 = _extract_image_candidates(match, key)
                idx = _find_ref_line_index(ref_lines, c0, c1)
                if idx is not None:
                    appended_map[idx] = appended_str

            total_pairs += 1

    thresholds = [5, 10, 20]
    aucs = [100.0 * yy for yy in pose_auc(pose_errors, thresholds)]
    n = float(len(pose_errors)) if pose_errors else 1.0
    Acc5 = np.sum(np.array(pose_errors) < 5.0) / n * 100 if pose_errors else 0.0
    Acc10 = np.sum(np.array(pose_errors) < 10.0) / n * 100 if pose_errors else 0.0
    Acc15 = np.sum(np.array(pose_errors) < 15.0) / n * 100 if pose_errors else 0.0
    Acc20 = np.sum(np.array(pose_errors) < 20.0) / n * 100 if pose_errors else 0.0
    mAP10 = np.mean([Acc5, Acc10])
    mAP20 = np.mean([Acc5, Acc10, Acc15, Acc20])
    percentage_of_correct = float(np.mean(percentage_of_correct_points) * 100) if percentage_of_correct_points else 0.0
    matching_score = float(np.mean(confs) * 100) if confs else 0.0

    summary = {
        'num_pairs': total_pairs,
        'AUC@5': aucs[0], 'AUC@10': aucs[1], 'AUC@20': aucs[2],
        'Acc@5': Acc5, 'Acc@10': Acc10, 'Acc@15': Acc15, 'Acc@20': Acc20,
        'mAP@5': Acc5, 'mAP@10': mAP10, 'mAP@20': mAP20,
        'percentage_of_correct': percentage_of_correct,
        'matching_score': matching_score,
    }

    return pair_lines, appended_map, summary


def _write_method_eval(method_dir, ref_lines, pair_lines, appended_map, summary):
    out_eval_path = osp.join(method_dir, "_eval.txt")
    summary_block = _summary_lines(summary)

    if ref_lines is not None:
        _write_ref_with_evals(ref_lines, appended_map, out_eval_path)
        with open(out_eval_path, 'a') as f:
            f.writelines(summary_block)
    else:
        with open(out_eval_path, 'w') as f:
            for line in pair_lines:
                f.write(line + "\n")
            f.writelines(summary_block)

    return out_eval_path


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate methods under a directory and generate _eval.txt per method plus a summary CSV.")
    parser.add_argument('--dir', type=str, default='outputs/eval_500_g', help='Directory of a single method or a directory containing multiple methods')
    parser.add_argument('--ref', type=str, default='data/megadepth_scale_2.txt', help='Optional reference pairs file (e.g., data/megadepth_scale_2.txt) to append eval numbers onto its lines for each method')
    args = parser.parse_args()

    in_dir = args.dir
    if not osp.isdir(in_dir):
        print(f"Given --dir is not a directory: {in_dir}")
        return

    ref_lines = _load_reference_lines(args.ref) if args.ref else None

    methods = _collect_methods(in_dir)
    if not methods:
        print(f"No valid methods found under {in_dir}")
        return

    # Prepare summary CSV in the provided directory
    summary_csv = osp.join(in_dir, 'methods_summary.csv')
    header = ['method', 'num_pairs', 'AUC@5', 'AUC@10', 'AUC@20', 'Acc@5', 'Acc@10', 'Acc@15', 'Acc@20',
              'mAP@5', 'mAP@10', 'mAP@20', 'percentage_of_correct', 'matching_score']

    rows = []
    for method_dir in methods:
        method_name = osp.basename(method_dir.rstrip('/'))
        pair_lines, appended_map, summary = _evaluate_method(method_dir, ref_lines=ref_lines)
        out_eval_path = _write_method_eval(method_dir, ref_lines, pair_lines, appended_map, summary)
        print(f"Wrote {out_eval_path}")

        row = [method_name, summary['num_pairs'], f"{summary['AUC@5']:.2f}", f"{summary['AUC@10']:.2f}", f"{summary['AUC@20']:.2f}",
               f"{summary['Acc@5']:.2f}", f"{summary['Acc@10']:.2f}", f"{summary['Acc@15']:.2f}", f"{summary['Acc@20']:.2f}",
               f"{summary['mAP@5']:.2f}", f"{summary['mAP@10']:.2f}", f"{summary['mAP@20']:.2f}",
               f"{summary['percentage_of_correct']:.2f}", f"{summary['matching_score']:.2f}"]
        rows.append(row)

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote summary CSV: {summary_csv}")


if __name__ == '__main__':
    main()