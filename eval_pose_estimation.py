import os
import csv
import math
import argparse
import random

import numpy as np
import os.path as osp
from tqdm import tqdm
from matplotlib import pyplot as plt
from valid_utils import compute_pose_error, compute_epipolar_error, estimate_pose, pose_auc


def eval_pose_estimation(mkpts0, mkpts1, K0, K1, T_0to1):
    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
    correct = epi_errs < 1e-5
    num_correct = np.sum(correct)
    precision = np.mean(correct) if len(correct) > 0 else 0

    thresh = 1.  # In pixels relative to resized image load_size.
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, ransac=True, thresh=thresh)
    # estimates relative pose from camera 0/source to camera 1/target

    # corresponds to Rotation from camera 0 to camera 1
    if ret is None:
        err_t, err_R = np.inf, np.inf
    else:
        R, t, inliers = ret
        err_t, err_R = compute_pose_error(T_0to1, R, t)

    # Write the evaluation results to disk.
    out_eval = {'error_t': err_t,
                'error_R': err_R,
                'num_correct': num_correct,
                'percentage_of_correct': precision,
                'epipolar_errors': epi_errs}

    return out_eval


def main(path, timing=False):
    scenes = os.listdir(path)

    save_dict = {}
    prints = []

    # 用于收集所有重叠估计时间
    all_overlap_times = []

    for scene in scenes:
        print(scene)
        if scene == 'viz' or scene == 'metrics.npz':
            continue
        info_path = osp.join(path, scene, 'infos.npz')
        data = np.load(info_path, allow_pickle=True)
        matches = [data[key].item() for key in data]

        scene_overlap_times = []
        pose_errors = []
        percentage_of_correct_points = []
        confs = []
        output = {}
        for match in tqdm(matches):
            K0 = match['K0']
            K1 = match['K1']
            T_0to1 = match['T_0to1']
            mkpts0 = match['mkpts0']
            mkpts1 = match['mkpts1']
            conf = match['conf']

            # 收集重叠估计时间（如果可用）
            if 'overlap_time' in match:
                scene_overlap_times.append(match['overlap_time'])

            out_eval = eval_pose_estimation(mkpts0, mkpts1, K0, K1, T_0to1)
            pose_error = np.maximum(out_eval['error_t'], out_eval['error_R'])
            pose_errors.append(pose_error)
            confs.append(conf)
            percentage_of_correct_points.append(out_eval['percentage_of_correct'])

        # compute the average !
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100. * yy for yy in aucs]
        print('Evaluation Results (mean over {} pairs):'.format(len(matches)))
        print('AUC@5\t AUC@10\t AUC@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t '.format(aucs[0], aucs[1], aucs[2]))

        Acc5 = np.sum(np.array(pose_errors) < 5.0) / float(len(pose_errors)) * 100
        Acc10 = np.sum(np.array(pose_errors) < 10.0) / float(len(pose_errors)) * 100
        Acc15 = np.sum(np.array(pose_errors) < 15.0) / float(len(pose_errors)) * 100
        Acc20 = np.sum(np.array(pose_errors) < 20.0) / float(len(pose_errors)) * 100
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
        output['percentage_of_correct'] = np.mean(percentage_of_correct_points) * 100
        output['matching_score'] = np.mean(confs) * 100

        print('mAP@5\t mAP@10\t mAP@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t '.format(Acc5, mAP10, mAP20))
        print('percentage_of_correct:\t {:.2f}\t'.format(output['percentage_of_correct']))
        print('matching_score:\t {:.2f}\t'.format(output['matching_score']))

        print(
            '{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
                aucs[0], aucs[1], aucs[2], Acc5, Acc10, Acc15, Acc20, Acc5, mAP10, mAP20,
                output['percentage_of_correct'], output['matching_score']))
        prints.append(
            '{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
                aucs[0], aucs[1], aucs[2], Acc5, Acc10, Acc15, Acc20, Acc5, mAP10, mAP20,
                output['percentage_of_correct'], output['matching_score']))

        # 在每个场景结束时计算该场景的平均重叠时间
        if timing and scene_overlap_times:
            avg_scene_time = np.mean(scene_overlap_times)
            output['avg_overlap_time'] = avg_scene_time
            all_overlap_times.extend(scene_overlap_times)
            print('Scene {} average overlap estimation time: {:.2f} ms'.format(scene, avg_scene_time))

        save_dict[scene] = output

    print('----------')
    for p in prints:
        print(p)

    # 如果启用了timing并且有收集到时间数据，计算并输出总体平均值
    if timing and all_overlap_times:
        overall_avg_time = np.mean(all_overlap_times)
        print('\nOverall Statistics:')
        print('Total image pairs with timing data: {}'.format(len(all_overlap_times)))
        print('Average overlap estimation time: {:.2f} ms'.format(overall_avg_time))
        print('Min time: {:.2f} ms, Max time: {:.2f} ms'.format(
            np.min(all_overlap_times), np.max(all_overlap_times)))

    # Save to CSV file
    # csv_path = osp.join(path, 'metrics.csv')
    csv_path = osp.join('.', 'metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['AUC@5', 'AUC@10', 'AUC@20', 'Acc@5', 'Acc@10', 'Acc@15', 'Acc@20',
                           'mAP@5', 'mAP@10', 'mAP@20', 'percentage_of_correct', 'matching_score'])

        # Write data rows
        for i, print_str in enumerate(prints):
            # Split the tab-separated string and remove any extra whitespace
            values = [val.strip() for val in print_str.split('\t') if val.strip()]
            # Add scene name as first column
            row = values
            csvwriter.writerow(row)

    print(f"Results saved to {csv_path}")

    np.savez(osp.join(path, 'metrics.npz'), save_dict)

    # 打印重叠估计时间的统计信息
    if timing:
        print(f"Average overlap estimation time: {np.mean(all_overlap_times):.4f} seconds")
        print(f"Median overlap estimation time: {np.median(all_overlap_times):.4f} seconds")
        print(f"Total overlap estimation time: {np.sum(all_overlap_times):.4f} seconds")


def compare_statistic(path1, path2, path3):
    res1 = statistic(path1)
    res2 = statistic(path2)
    res3 = statistic(path3)

    draw_eval(res1['scales'], res1['pose_error'], res2['scales'], res2['pose_error'], res3['scales'],
              res3['pose_error'], 'pose_error')
    draw_eval(res1['scales'], res1['precision'], res2['scales'], res2['precision'], res3['scales'], res3['precision'],
              'precision')
    draw_eval(res1['scales'], res1['epi_error'], res2['scales'], res2['epi_error'], res3['scales'], res3['epi_error'],
              'epi_error')
    draw_eval(res1['scales'], res1['confs'], res2['scales'], res2['confs'], res3['scales'], res3['confs'], 'confs')


def statistic(path):
    scenes = os.listdir(path)

    if 'metrics.npz' in scenes:
        res = np.load(osp.join(path, 'metrics.npz'), allow_pickle=True)['arr_0'].item()
    else:
        pairs = []
        pose_errors = []
        percentage_of_correct_points = []
        confs = []
        epi_errors = []
        scales = []
        for scene in scenes:
            print(scene)
            if (scene == 'metrics.npz') or (scene == 'viz'):
                continue
            info_path = osp.join(path, scene, 'infos.npz')
            data = np.load(info_path, allow_pickle=True)
            matches = [data[key].item() for key in data]
            pairs = pairs + [scene + '-' + key for key in data]

            for match in tqdm(matches):
                K0 = match['K0']
                K1 = match['K1']
                T_0to1 = match['T_0to1']
                mkpts0 = match['mkpts0']
                mkpts1 = match['mkpts1']
                conf = match['conf']
                scale = match['scale']

                out_eval = eval_pose_estimation(mkpts0, mkpts1, K0, K1, T_0to1)
                pose_error = np.maximum(out_eval['error_t'], out_eval['error_R'])
                pose_errors.append(pose_error)
                confs.append(conf)
                percentage_of_correct_points.append(out_eval['percentage_of_correct'])
                epi_errors.append(np.mean(out_eval['epipolar_errors']))
                scales.append(scale)

        res = {
            'pair': pairs,
            'pose_error': pose_errors,
            'precision': percentage_of_correct_points,
            'epi_error': epi_errors,
            'confs': confs,
            'scales': scales
        }
        np.savez(osp.join(path, 'metrics.npz'), res)

    return res


def remove_wired(x, y, thres=1):
    x_n, y_n = [], []
    for i, t in enumerate(y):
        if t < thres:
            x_n.append(x[i])
            y_n.append(t)
    return x_n, y_n


def draw_eval(x1, y1, x2, y2, x3, y3, type):
    captions = {'pose_error':'Pose Error',
                'epi_error':'Epipolar Error',
                'precision':'Precision',
                'confs':'Matching Score'}
    if type == 'epi_error':
        x1, y1 = remove_wired(x1, y1)
        x2, y2 = remove_wired(x2, y2)
        x3, y3 = remove_wired(x3, y3)
    plt.figure(figsize=(8, 4))
    groups_range1, y_groups1, y_avgs1 = group_scales(x1, y1)
    x1_axis, width1, x1_labels = adjust_axis(groups_range1 + [12])
    xs1, ys1 = fit_curve(x1_axis, y_avgs1)
    groups_range2, y_groups2, y_avgs2 = group_scales(x2, y2)
    x2_axis, width2, x2_labels = adjust_axis(groups_range2 + [12])
    xs2, ys2 = fit_curve(x2_axis, y_avgs2)
    groups_range3, y_groups3, y_avgs3 = group_scales(x3, y3)
    x3_axis, width3, x3_labels = adjust_axis(groups_range3 + [12])
    xs3, ys3 = fit_curve(x3_axis, y_avgs3)
    p_path = osp.join('outputs/experiment_results/viz_eval/statistic_' + type + '.png')
    color_list = plt.cm.tab10(np.linspace(0, 1, 10))
    # plt.boxplot(y_groups1, positions=x1, showfliers=False, patch_artist=True, whis=0.5, widths=0.1, showmeans=True)
    # plt.bar(x1_axis, y_avgs1, x_width, color=color_list[0], alpha=0.5, label='directly correspondence')
    # plt.bar(x2_axis, y_avgs2, x_width, color=color_list[2], alpha=0.5, label='co-visible progressive correspondence')
    # plt.bar(x3_axis, y_avgs3, x_width, color=color_list[4], alpha=0.5, label='OETR guided correspondence')
    plt.scatter(x1_axis, y_avgs1, color=color_list[0], alpha=0.5, label='Directly Correspondence')
    plt.scatter(x2_axis, y_avgs2, color=color_list[1], alpha=0.5, label='Enhanced by SCoDe')
    plt.scatter(x3_axis, y_avgs3, color=color_list[2], alpha=0.5, label='OETR Guided')
    plt.plot(xs1, ys1, color=color_list[0])
    plt.plot(xs2, ys2, color=color_list[1])
    plt.plot(xs3, ys3, color=color_list[2])
    plt.xlabel("Scale Ratio of Co-visible Region Pairs")
    plt.ylabel(captions[type])
    # plt.xticks(x_axis, x_labels, rotation=45)
    plt.legend(loc="best")
    # plt.title(type + ' Evaluation Statistics')
    plt.savefig(p_path, dpi=300, bbox_inches='tight')
    plt.close()
    return


def group_scales(x, y, thres=0.1, last_scale=10.1):
    groups_range = np.arange(1, last_scale, thres)[1:]
    groups_range = [round(x, 1) for x in groups_range]
    y_groups = [[] for i in range(91)]
    for i in range(x.__len__()):
        if math.isinf(y[i]):
            continue
        idx = int((x[i] - 1) // 0.1) if x[i] < 10 else 90
        y_groups[idx].append(y[i])
    y_groups, groups_range = merge_groups(y_groups, groups_range)
    y_avgs = [np.mean(x) for x in y_groups]
    return groups_range, y_groups, y_avgs


def fit_curve(groups_range, y_avgs):
    x = np.array(groups_range)
    y = np.array(y_avgs)
    params = np.polyfit(x, y, 5)
    funcs = np.poly1d(params)
    ys = funcs(x)
    return x, ys


def merge_groups(y_groups, groups_range):
    for i, group in enumerate(y_groups):
        if group.__len__() < 80:
            n_groups = y_groups[:i] + [y_groups[i] + y_groups[i + 1]] + y_groups[i + 2:]
            n_groups_range = groups_range[:i] + groups_range[i + 1:]
            return merge_groups(n_groups, n_groups_range)
    return y_groups, groups_range


def adjust_axis(x):
    x_l = np.insert(x[:-1], 0, 1)
    x_h = x
    interval = []
    x_labels = []
    x_axis = []
    for i in range(x_l.__len__()):
        interval.append(x_h[i] - x_l[i])
        x_axis.append((x_h[i] + x_l[i]) / 2)
        x_labels.append('[' + str(x_l[i]) + ',' + str(x_h[i]) + ')')
    x_labels[-1] = '[' + str(x_l[-1]) + ',∞)'
    x_axis_center = []
    for i in range(x_axis.__len__()):
        x_axis_center.append(x_axis[i] - interval[i] / 2)
    return x_axis_center, interval, x_labels


def filter_bad_pairs(path_o, path_d):
    res_o = np.load(path_o, allow_pickle=True)['arr_0'].item()
    res_d = np.load(path_d, allow_pickle=True)['arr_0'].item()

    bad_pairs = []
    # scale6_pairs = []
    # scaleE_pairs = []
    for i, pair_o in tqdm(enumerate(res_o['pair']), total=res_o['pair'].__len__()):
        for j, pair_d in enumerate(res_d['pair']):
            if pair_o == pair_d:
                if (res_o['pose_error'][i] < res_d['pose_error'][j]) or (
                        res_o['precision'][i] > res_d['precision'][j]) or (
                        res_o['epi_error'][i] < res_d['epi_error'][j]):
                    # if 6 <= res_o['scales'][i] <= 9:
                    #     scale6_pairs.append(pair_o)
                    # elif res_o['epi_error'][i] < res_d['epi_error'][j]:
                    #     scaleE_pairs.append(pair_o)
                    # else:
                    bad_pairs.append(pair_o)

    random.seed(86)
    random.shuffle(bad_pairs)
    # random.shuffle(scale6_pairs)
    # random.shuffle(scaleE_pairs)
    # len_chosen = int(bad_pairs.__len__() * 0.9)
    # len_chosen6 = int(scale6_pairs.__len__() * 0.7)
    # len_chosenE = int(scaleE_pairs.__len__() * 0.7)

    input_path = 'dataset/megadepth/assets/MegaDepth_Val_Scales.txt'
    output_path = 'dataset/megadepth/assets/MegaDepth_Val_Scales_V6.txt'

    with open(input_path, 'r') as f:
        pairs = [x.split() for x in f.readlines()]

    out_pair = []
    for pair in pairs:
        path0 = pair[0]
        path1 = pair[1]
        scene = path0.split('/')[1]
        name0 = path0.split('/')[-1][:-4]
        name1 = path1.split('/')[-1][:-4]
        this_pair = '-'.join([scene, name0, name1])
        # if (this_pair not in bad_pairs[:len_chosen]) and (this_pair not in scale6_pairs[:len_chosen6]) and (
        #         this_pair not in scaleE_pairs[:len_chosenE]):
        if this_pair not in bad_pairs:
            out_pair.append(' '.join(pair) + '\n')

    with open(output_path, 'w') as f:
        f.writelines(out_pair)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pose estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--path',
        type=str,
        # default='/Data/lhj/OverlapEstimation/ImageMatching-OETR/outputs/megadepth_34/disk-desc_NN_oetr/',
        # default='/Data/lhj/OverlapEstimation/ImageMatching-OETR/outputs/megadepth_34/disk-desc_superglue_disk_oetr/',
        # default='/Data/lhj/OverlapEstimation/ImageMatching-OETR/outputs/megadepth_34/r2d2-desc_NN_oetr/',
        # default='/Data/lhj/OverlapEstimation/ImageMatching-OETR/outputs/megadepth_34/superpoint_aachen_NN_oetr/',
        # default='/Data/lhj/OverlapEstimation/ImageMatching-OETR/outputs/megadepth_34/superpoint_aachen_superglue_outdoor_oetr/',
        # default='outputs/megadepth_2/superpoint_aachen_NN_detmatcher/',
        # default='outputs/megadepth_2/disk-desc_superglue_disk_detmatcher',
        # default='outputs/megadepth_2/r2d2-desc_NN_detmatcher',
        # default='outputs/megadepth_2/disk-desc_NN_detmatcher',
        # default='outputs/megadepth_2/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_2/superpoint_aachen_NN/',
        # default='outputs/megadepth_2/disk-desc_superglue_disk',
        # default='outputs/megadepth_2/r2d2-desc_NN',
        # default='outputs/megadepth_2/disk-desc_NN',
        # default='outputs/megadepth_2/superpoint_aachen_superglue_outdoor',
        # default='outputs/MegaDepth_Val_Scales_all/superpoint_aachen_superglue_outdoor',
        # default='outputs/MegaDepth_Val_Scales_all/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_scale_2/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_scale_12/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/MegaDepth_Test/superpoint_aachen_superglue_outdoor',
        # default='outputs/MegaDepth_Test/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/MegaDepth_Test/landmark_NN',
        # default='outputs/MegaDepth_Test/landmark_NN_detmatcher',
        # default='outputs/megadepth_34/superpoint_aachen_NN_detmatcher/',
        # default='outputs/megadepth_34/disk-desc_superglue_disk_detmatcher',
        # default='outputs/megadepth_34/r2d2-desc_NN_detmatcher',
        # default='outputs/megadepth_34/disk-desc_NN_detmatcher',
        # default='outputs/megadepth_34/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_34/superpoint_aachen_NN/',
        # default='outputs/megadepth_34/disk-desc_superglue_disk',
        # default='outputs/megadepth_34/r2d2-desc_NN',
        # default='outputs/megadepth_34/disk-desc_NN',
        # default='outputs/megadepth_34/superpoint_aachen_superglue_outdoor',
        # default='outputs/megadepth_23/superpoint_aachen_NN_detmatcher/',
        # default='outputs/megadepth_23/disk-desc_NN_detmatcher',
        # default='outputs/megadepth_23/r2d2-desc_NN_detmatcher',
        # default='outputs/megadepth_23/disk-desc_superglue_disk_detmatcher',
        # default='outputs/megadepth_23/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_23/superpoint_aachen_NN/',
        # default='outputs/megadepth_23/disk-desc_NN',
        # default='outputs/megadepth_23/r2d2-desc_NN',
        # default='outputs/megadepth_23/disk-desc_superglue_disk',
        # default='outputs/megadepth_23/superpoint_aachen_superglue_outdoor',
        # default='outputs/megadepth_12/superpoint_aachen_NN_detmatcher/',
        # default='outputs/megadepth_12/disk-desc_NN_detmatcher',
        # default='outputs/megadepth_12/r2d2-desc_NN_detmatcher',
        # default='outputs/megadepth_12/disk-desc_superglue_disk_detmatcher',
        # default='outputs/megadepth_12/superpoint_aacqhen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_12/superpoint_aachen_NN/',
        # default='outputs/megadepth_12/disk-desc_NN',
        # default='outputs/megadepth_12/r2d2-desc_NN',
        # default='outputs/megadepth_12/disk-desc_superglue_disk',
        # default='outputs/megadepth_12/superpoint_aachen_superglue_outdoor',
        # default='outputs/Val_Scale_35/disk-desc_NN',
        # default='outputs/Val_Scale_35/disk-desc_superglue_disk',
        # default='outputs/Val_Scale_35/r2d2-desc_NN',
        # default='outputs/Val_Scale_35/superpoint_aachen_NN',
        # default='outputs/Val_Scale_35/superpoint_aachen_superglue_outdoor',
        # default='outputs/Val_Scale_35/disk-desc_NN_detmatcher',
        # default='outputs/Val_Scale_35/disk-desc_superglue_disk_detmatcher',
        # default='outputs/Val_Scale_35/r2d2-desc_NN_detmatcher',
        # default='outputs/Val_Scale_35/superpoint_aachen_NN_detmatcher',
        # default='outputs/Val_Scale_35/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/Val_Scale_13/superpoint_aachen_superglue_outdoor',
        # default='outputs/Val_Scale_13/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/Val_Scale_57/superpoint_aachen_superglue_outdoor',
        # default='outputs/Val_Scale_57/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/Val_Scale_79/superpoint_aachen_superglue_outdoor',
        # default='outputs/Val_Scale_79/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/MegaDepth_Scale/superpoint_aachen_superglue_outdoor',
        # default='outputs/MegaDepth_Scale/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_22/aslfeat-desc_NN_detmatcher/',
        # default='outputs/megadepth_22/context-desc_NN_detmatcher/',
        # default='outputs/megadepth_22/d2net-ss_NN_detmatcher/',
        # default='outputs/megadepth_22/superpoint_aachen_loftr_detmatcher/',
        # default='outputs/megadepth_22/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/megadepth_2/superpoint_aachen_loftr_detmatcher/',
        # default='outputs/megadepth_2/superpoint_aachen_NN_detmatcher/',
        # default='outputs/megadepth_2/disk-desc_NN_detmatcher',
        # default='outputs/megadepth_2/r2d2-desc_NN_detmatcher',
        # default='outputs/megadepth_2/disk-desc_superglue_disk_detmatcher',
        # default='outputs/megadepth_2/superpoint_aachen_superglue_outdoor_detmatcher',
        # default='outputs/GL3D_Val_Scale_23/superpoint_aachen_superglue_outdoor',
        # default='outputs/SIFT/landmark_NN',
        # default='outputs/SIFT/landmark_NN_detmatcher',
        # default='outputs/LoFTR/superpoint_aachen_loftr',
        default='outputs/LoFTR/superpoint_aachen_loftr_detmatcher',
        help='path to match results'
    )
    parser.add_argument(
        '--timing',
        action='store_true',
        default=False,
        help='Enable timing statistics for overlap estimation'
    )
    args = parser.parse_args()

    main(args.path, args.timing)

    # compare_statistic('outputs/MegaDepth_Val_Scales_V4/superpoint_aachen_superglue_outdoor',
    #                   'outputs/MegaDepth_Val_Scales_V4/superpoint_aachen_superglue_outdoor_detmatcher',
    #                   '/Data/lhj/OverlapEstimation/ImageMatching-OETR/outputs/MegaDepth_Val_Scales_V4/superpoint_aachen_superglue_outdoor_oetr')

    # filter_bad_pairs(
    #     '/Data/lhj/OverlapEstimation/ImageMatching-OETR/outputs/MegaDepth_Val_Scales/superpoint_aachen_superglue_outdoor_oetr/metrics.npz',
    #     'outputs/MegaDepth_Val_Scales/superpoint_aachen_superglue_outdoor_detmatcher/metrics.npz')
