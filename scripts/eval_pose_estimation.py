import os
import csv
import argparse

import numpy as np
import os.path as osp
from tqdm import tqdm
from scripts.valid_utils import compute_pose_error, compute_epipolar_error, estimate_pose, pose_auc


def eval_pose_estimation(mkpts0, mkpts1, K0, K1, T_0to1):
    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
    correct = epi_errs < 1e-5
    num_correct = np.sum(correct)
    precision = np.mean(correct) if len(correct) > 0 else 0

    # RANSAC threshold in pixels for resized images.
    thresh = 1.
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, ransac=True, thresh=thresh)

    # Relative pose from camera 0 (source) to camera 1 (target).
    if ret is None:
        err_t, err_R = np.inf, np.inf
    else:
        R, t, inliers = ret
        err_t, err_R = compute_pose_error(T_0to1, R, t)

    out_eval = {
        'error_t': err_t,
        'error_R': err_R,
        'num_correct': num_correct,
        'percentage_of_correct': precision,
        'epipolar_errors': epi_errs
    }
    return out_eval


def _load_reference_lines(ref_path):
    """Load reference megadepth-like file lines. Return list or None if missing."""
    if osp.exists(ref_path):
        with open(ref_path, 'r') as f:
            return f.readlines()
    return None


def _find_ref_line_index(ref_lines, cands0, cands1):
    """
    Try to find an index in ref_lines that contains one candidate from cands0 and one from cands1.
    Returns index or None.
    """
    if ref_lines is None:
        return None
    for i, line in enumerate(ref_lines):
        for a in cands0:
            if a == '':
                continue
            for b in cands1:
                if b == '':
                    continue
                if (a in line) and (b in line):
                    return i
    return None


def _extract_image_candidates(match, key):
    """
    Build lists of possible image basename candidates (with or without .jpg) from match dict and key.
    Returns (cands0, cands1) where each is a list of strings.
    """
    c0 = []
    c1 = []

    # common key names used in different pipelines
    possible_keys = [
        ('path0', 'path1'),
        ('path_0', 'path_1'),
        ('img0', 'img1'),
        ('image0', 'image1'),
        ('name0', 'name1'),
        ('file0', 'file1'),
    ]
    for k0, k1 in possible_keys:
        if k0 in match and k1 in match:
            v0 = str(match[k0])
            v1 = str(match[k1])
            # basename
            b0 = osp.basename(v0)
            b1 = osp.basename(v1)
            c0.append(b0)
            c1.append(b1)
            # with/without .jpg variants
            if not b0.lower().endswith('.jpg'):
                c0.append(b0 + '.jpg')
            if not b1.lower().endswith('.jpg'):
                c1.append(b1 + '.jpg')

    # also try direct name0/name1 if present but without extension
    if 'name0' in match and 'name1' in match:
        n0 = str(match['name0'])
        n1 = str(match['name1'])
        c0.append(n0)
        c1.append(n1)
        if not n0.lower().endswith('.jpg'):
            c0.append(n0 + '.jpg')
        if not n1.lower().endswith('.jpg'):
            c1.append(n1 + '.jpg')

    # Try to parse the info key if it encodes pair (e.g., "imgA.jpg-imgB.jpg" or "imgA-imgB")
    if isinstance(key, str) and '-' in key:
        parts = key.split('-')
        if len(parts) >= 2:
            p0 = parts[0]
            p1 = parts[1]
            c0.append(p0)
            c1.append(p1)
            if not p0.lower().endswith('.jpg'):
                c0.append(p0 + '.jpg')
            if not p1.lower().endswith('.jpg'):
                c1.append(p1 + '.jpg')

    # final fallbacks: empty strings will be ignored by finder
    if not c0:
        c0 = ['']
    if not c1:
        c1 = ['']

    # deduplicate
    c0 = list(dict.fromkeys(c0))
    c1 = list(dict.fromkeys(c1))
    return c0, c1


def _write_ref_with_evals(ref_lines, appended_map, out_path):
    """
    Write ref_lines to out_path, appending appended_map[i] to ref_lines[i] if present.
    appended_map: dict mapping line_index -> appended string (without newline).
    """
    if ref_lines is None:
        return
    with open(out_path, 'w') as f:
        for i, line in enumerate(ref_lines):
            line_stripped = line.rstrip('\n\r')
            if i in appended_map:
                f.write(line_stripped + ' ' + appended_map[i] + '\n')
            else:
                f.write(line)
    return


def _mean_epipolar_error(out_eval):
    """Return mean epipolar error or NaN when unavailable."""
    epi = out_eval.get('epipolar_errors')
    if epi is None or np.size(epi) == 0:
        return np.nan
    return float(np.mean(epi))


def _format_appended_eval(out_eval, conf):
    """Build the appended per-pair eval string for reference-line output."""
    return "{:.6f} {:.6f} {} {:.6f} {:.6f} {:.6f}".format(
        float(out_eval.get('error_t', np.nan)),
        float(out_eval.get('error_R', np.nan)),
        int(out_eval.get('num_correct', 0)),
        float(out_eval.get('percentage_of_correct', 0.0)),
        _mean_epipolar_error(out_eval),
        float(conf) if conf is not None else np.nan
    )


def _format_scene_row(aucs, acc5, acc10, acc15, acc20, map10, map20, pct_correct, matching_score):
    """Serialize one scene metric row for printing/CSV."""
    return [
        f"{aucs[0]:.2f}", f"{aucs[1]:.2f}", f"{aucs[2]:.2f}",
        f"{acc5:.2f}", f"{acc10:.2f}", f"{acc15:.2f}", f"{acc20:.2f}",
        f"{acc5:.2f}", f"{map10:.2f}", f"{map20:.2f}",
        f"{pct_correct:.2f}", f"{matching_score:.2f}"
    ]


def main(path, timing=False, ref_path='data/megadepth_scale_2.txt'):
    scenes = os.listdir(path)
    skip_entries = {'viz', 'viz_debug', 'metrics.npz'}

    save_dict = {}
    print_rows = []
    csv_rows = []
    all_overlap_times = []

    # Optional reference file for writing per-pair appended evaluations.
    ref_lines = _load_reference_lines(ref_path)
    appended_map = {}

    for scene in scenes:
        print(scene)
        if scene in skip_entries:
            continue

        info_path = osp.join(path, scene, 'infos.npz')
        if not osp.exists(info_path):
            continue

        with np.load(info_path, allow_pickle=True) as data:
            matches = [(k, data[k].item()) for k in data.files]

        scene_overlap_times = []
        pose_errors = []
        percentage_of_correct_points = []
        confs = []
        output = {}

        for key, match in tqdm(matches):
            K0 = match['K0']
            K1 = match['K1']
            T_0to1 = match['T_0to1']
            mkpts0 = match['mkpts0']
            mkpts1 = match['mkpts1']
            conf = match.get('conf', np.nan)

            if 'overlap_time' in match:
                scene_overlap_times.append(match['overlap_time'])

            out_eval = eval_pose_estimation(mkpts0, mkpts1, K0, K1, T_0to1)
            pose_errors.append(np.maximum(out_eval['error_t'], out_eval['error_R']))
            confs.append(conf if conf is not None else np.nan)
            percentage_of_correct_points.append(out_eval['percentage_of_correct'])

            if ref_lines is not None:
                c0, c1 = _extract_image_candidates(match, key)
                idx = _find_ref_line_index(ref_lines, c0, c1)
                if idx is not None:
                    appended_map[idx] = _format_appended_eval(out_eval, conf)

        thresholds = [5, 10, 20]
        aucs = [100. * yy for yy in pose_auc(pose_errors, thresholds)]
        print('Evaluation Results (mean over {} pairs):'.format(len(matches)))
        print('AUC@5\t AUC@10\t AUC@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t '.format(aucs[0], aucs[1], aucs[2]))

        denom = float(len(pose_errors)) if pose_errors else 1.0
        Acc5 = np.sum(np.array(pose_errors) < 5.0) / denom * 100 if pose_errors else 0.0
        Acc10 = np.sum(np.array(pose_errors) < 10.0) / denom * 100 if pose_errors else 0.0
        Acc15 = np.sum(np.array(pose_errors) < 15.0) / denom * 100 if pose_errors else 0.0
        Acc20 = np.sum(np.array(pose_errors) < 20.0) / denom * 100 if pose_errors else 0.0
        mAP10 = np.mean([Acc5, Acc10])
        mAP20 = np.mean([Acc5, Acc10, Acc15, Acc20])

        output['AUC@5'] = aucs[0]
        output['AUC@10'] = aucs[1]
        output['AUC@20'] = aucs[2]
        output['Acc@5'] = Acc5
        output['Acc@10'] = Acc10
        output['Acc@15'] = Acc15
        output['Acc@20'] = Acc20

        print('Acc@5\t Acc@10\t Acc@15\t Acc@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(Acc5, Acc10, Acc15, Acc20))

        output['mAP@5'] = Acc5
        output['mAP@10'] = mAP10
        output['mAP@20'] = mAP20
        output['percentage_of_correct'] = (np.mean(percentage_of_correct_points) * 100) if percentage_of_correct_points else 0.0
        output['matching_score'] = (np.mean(confs) * 100) if confs else 0.0

        print('mAP@5\t mAP@10\t mAP@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t '.format(Acc5, mAP10, mAP20))
        print('percentage_of_correct:\t {:.2f}\t'.format(output['percentage_of_correct']))
        print('matching_score:\t {:.2f}\t'.format(output['matching_score']))

        row = _format_scene_row(
            aucs, Acc5, Acc10, Acc15, Acc20, mAP10, mAP20,
            output['percentage_of_correct'], output['matching_score']
        )
        print('\t '.join(row))
        print_rows.append('\t '.join(row))
        csv_rows.append(row)

        if timing and scene_overlap_times:
            avg_scene_time = np.mean(scene_overlap_times)
            output['avg_scene_time'] = avg_scene_time
            all_overlap_times.extend(scene_overlap_times)
            print('Scene {} average overlap estimation time: {:.2f} ms'.format(scene, avg_scene_time))

        save_dict[scene] = output

    print('----------')
    for p in print_rows:
        print(p)

    if timing and all_overlap_times:
        overall_avg_time = np.mean(all_overlap_times)
        print('\nOverall Statistics:')
        print('Total image pairs with timing data: {}'.format(len(all_overlap_times)))
        print('Average overlap estimation time: {:.2f} ms'.format(overall_avg_time))
        print('Min time: {:.2f} ms, Max time: {:.2f} ms'.format(np.min(all_overlap_times), np.max(all_overlap_times)))

    csv_path = osp.join('.', 'metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['AUC@5', 'AUC@10', 'AUC@20', 'Acc@5', 'Acc@10', 'Acc@15', 'Acc@20',
                  'mAP@5', 'mAP@10', 'mAP@20', 'percentage_of_correct', 'matching_score']
        csvwriter.writerow(header)

        for row in csv_rows:
            csvwriter.writerow(row)

        if csv_rows:
            arr = np.array(csv_rows, dtype=float)
            col_means = np.mean(arr, axis=0)
            csvwriter.writerow([f"{m:.2f}" for m in col_means.tolist()])

    print(f"Results saved to {csv_path}")
    np.savez(osp.join(path, 'metrics.npz'), save_dict)

    if ref_lines is not None:
        out_ref_path = ref_path.replace('.txt', '_eval.txt')
        _write_ref_with_evals(ref_lines, appended_map, out_ref_path)
        print(f"Per-pair evals appended and saved to {out_ref_path}")
    else:
        print("Reference file not found; skipped writing per-pair eval txt.")

    if timing and all_overlap_times:
        print(f"Average overlap estimation time: {np.mean(all_overlap_times):.4f} seconds")
        print(f"Median overlap estimation time: {np.median(all_overlap_times):.4f} seconds")
        print(f"Total overlap estimation time: {np.sum(all_overlap_times):.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pose estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--path',
        type=str,
        # default='outputs/eval/context-desc_NN_samatcher',
        # default='outputs/eval/d2net-ss_NN_samatcher',
        # default='outputs/eval/disk-desc_NN_samatcher',
        # default='outputs/eval/disk-desc_superglue_disk_samatcher',
        # default='outputs/eval/landmark_NN_samatcher',
        # default='outputs/eval/r2d2-desc_NN_samatcher',
        default='outputs/eval/superpoint_aachen_loftr_samatcher',
        # default='outputs/eval/superpoint_aachen_NN_samatcher',
        # default='outputs/eval/superpoint_aachen_superglue_outdoor_samatcher',
        help='path to match results'
    )
    parser.add_argument(
        '--timing',
        action='store_true',
        default=False,
        help='Enable timing statistics for overlap estimation'
    )
    parser.add_argument(
        '--ref',
        type=str,
        default='data/megadepth_scale_2.txt',
        help='Optional reference pair file for per-pair appended eval output'
    )
    args = parser.parse_args()

    main(args.path, args.timing, args.ref)
