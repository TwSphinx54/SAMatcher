import os
import h5py
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import cm as cm
import torch.nn.functional as F
from collections import defaultdict

from dloc.api import extract_process
from dloc.core.utils.base_model import dynamic_load
from dloc.core.match_features import preprocess_match_pipeline
from dloc.core.overlap_features import preprocess_overlap_pipeline
from dloc.core.utils.utils import make_matching_plot, tensor_overlap_crop, vis_aligned_image, pad_mask
from dloc.core import extract_features, extractors, match_features, matchers, overlap_features, overlaps

torch.set_grad_enabled(False)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Matching(torch.nn.Module):
    """Image Matching Frontend combining feature extractor and matcher."""

    def __init__(self, config=None, model_path=Path('weights/')):
        super(Matching, self).__init__()
        self.config = config

        # Initialize overlap estimator if specified
        if self.config['overlaper'] is not None:
            self.overlap = dynamic_load(overlaps,
                                        config['overlaper']['model']['name'])(
                config['overlaper']['model'], model_path)
        
        # Initialize feature extractor unless using direct matching
        if not self.config['direct']:
            self.extractor = dynamic_load(
                extractors, config['extractor']['model']['name'])(
                config['extractor']['model'], model_path)
        
        # Initialize matcher
        self.matcher = dynamic_load(matchers,
                                    config['matcher']['model']['name'])(
            config['matcher']['model'], model_path)
        
        self.extractor_name = self.config['extractor']['model']['name']
        self.matcher_name = self.config['matcher']['model']['name']
        self.size_divisor = 1

    def forward(self, data, with_overlap=False):
        """Run extractors and matchers on image pair.
        
        Args:
            data (dict): Input data with keys ['image0', 'image1'] and optional overlap data
            with_overlap (bool): Whether to use overlap estimation
            
        Returns:
            dict: Matching results including keypoints, matches, and scores
        """
        # Skip extractor for direct matching without overlap
        if self.config['direct'] and not with_overlap:
            return self.matcher(data)

        if with_overlap:
            pred = {}
            self.device = data['overlap_image1'].device
            
            # Ensure overlap scales are tensors
            assert isinstance(data['overlap_scales0'], tuple)
            assert isinstance(data['overlap_scales1'], tuple)
            data['overlap_scales0'] = torch.tensor(data['overlap_scales0'] +
                                                   data['overlap_scales0'],
                                                   device=self.device)
            data['overlap_scales1'] = torch.tensor(data['overlap_scales1'] +
                                                   data['overlap_scales1'],
                                                   device=self.device)
            
            # Measure overlap estimation time
            overlap_start_time = time.time()
            bbox0, bbox1, mask0, mask1 = self.overlap({
                'image0': data['overlap_image0'],
                'image1': data['overlap_image1']
            })
            overlap_time = (time.time() - overlap_start_time) * 1000  # Convert to milliseconds
            pred['overlap_time'] = torch.tensor([overlap_time], device=self.device)

            # Scale bounding boxes to original image coordinates
            bbox0 = bbox0 * data['overlap_scales0']
            bbox1 = bbox1 * data['overlap_scales1']

            r_h0, r_w0 = data['overlap_scales0'][1], data['overlap_scales0'][0]
            r_h1, r_w1 = data['overlap_scales1'][1], data['overlap_scales1'][0]
            new_h0 = int(mask0.shape[0] * r_h0)
            new_w0 = int(mask0.shape[1] * r_w0)
            new_h1 = int(mask1.shape[0] * r_h1)
            new_w1 = int(mask1.shape[1] * r_w1)
            scaled_mask0 = F.interpolate(mask0.unsqueeze(0).unsqueeze(0), size=(new_h0, new_w0), mode='bilinear',
                                         align_corners=False)
            scaled_mask1 = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(new_h1, new_w1), mode='bilinear',
                                         align_corners=False)
            threshold = 0.5
            b_mask0 = (scaled_mask0[0, 0].sigmoid() > threshold).float()
            b_mask1 = (scaled_mask1[0, 0].sigmoid() > threshold).float()
            scaled_mask0 = pad_mask(b_mask0, bbox0, outer_factor=50, threshold=threshold)
            scaled_mask1 = pad_mask(b_mask1, bbox1, outer_factor=50, threshold=threshold)

            # Calculate overlap region dimensions
            bw0, bh0 = (
                bbox0[0][2].int() - bbox0[0][0].int(),
                bbox0[0][3].int() - bbox0[0][1].int(),
            )
            bw1, bh1 = (
                bbox1[0][2].int() - bbox1[0][0].int(),
                bbox1[0][3].int() - bbox1[0][1].int(),
            )
            
            # Calculate overlap scale ratio
            overlap_scores = max(
                torch.floor_divide(bw0, bw1),
                torch.floor_divide(bh0, bh1),
                torch.floor_divide(bw1, bw0),
                torch.floor_divide(bh1, bh0),
            )

            # Process overlap regions if they meet size and ratio requirements
            if min(bw0, bh0, bw1, bh1) > 1 and (
                    (data['dataset_name'] == 'pragueparks-val'
                     and overlap_scores > 2.0)
                    or data['dataset_name'] != 'pragueparks-val'):
                
                # Adjust size divisor for LoFTR matcher
                if self.matcher_name == 'loftr':
                    self.size_divisor = 8

                if self.config['overlaper']['model']['name'] == 'samatcher':
                    crop_mask0, crop_mask1, _, _ = tensor_overlap_crop(
                        scaled_mask0, bbox0, scaled_mask1, bbox1,
                        self.extractor_name, self.size_divisor,
                    )
                
                # Crop overlap regions
                overlap0, overlap1, ratio0, ratio1 = tensor_overlap_crop(
                    data['image0'], bbox0, data['image1'], bbox1,
                    self.extractor_name, self.size_divisor,
                )

                pred.update({
                    'bbox0': bbox0,
                    'bbox1': bbox1,
                    'ratio0': torch.tensor(ratio0, device=bbox0.device),
                    'ratio1': torch.tensor(ratio1, device=bbox0.device),
                    'mask0': scaled_mask0,
                    'mask1': scaled_mask1,
                })
                
                # Direct matching on overlap regions
                if self.config['direct']:
                    matches = self.matcher({'image0': overlap0, 'image1': overlap1})
                    pred.update(matches)
                    return pred
                
                # Extract features from overlap regions
                if 'keypoints0' not in data:
                    pred0 = self.extractor({'image': overlap0})
                    if self.config['overlaper']['model']['name'] == 'samatcher':
                        valid_indices0 = [i for i, (x, y) in enumerate(pred0['keypoints'][0].int()) if
                                          crop_mask0.squeeze()[y, x]]
                        if valid_indices0.__len__() >= 3:
                            pred0['keypoints'] = [pred0['keypoints'][0][valid_indices0]]
                            pred0['scores'] = [pred0['scores'][0][valid_indices0]]
                            pred0['descriptors'] = [pred0['descriptors'][0][:, valid_indices0]]
                    pred.update(dict((k + '0', v) for k, v in pred0.items()))
                if 'keypoints1' not in data:
                    pred1 = self.extractor({'image': overlap1})
                    if self.config['overlaper']['model']['name'] == 'samatcher':
                        valid_indices1 = [i for i, (x, y) in enumerate(pred1['keypoints'][0].int()) if
                                          crop_mask1.squeeze()[y, x]]
                        if valid_indices1.__len__() >= 3:
                            pred1['keypoints'] = [pred1['keypoints'][0][valid_indices1]]
                            pred1['scores'] = [pred1['scores'][0][valid_indices1]]
                            pred1['descriptors'] = [pred1['descriptors'][0][:, valid_indices1]]
                    pred.update(dict((k + '1', v) for k, v in pred1.items()))

            else:
                # Use full images when overlap regions are too small
                pred.update({
                    'bbox0': torch.tensor([[0.0, 0.0, data['image0'].shape[3],
                                           data['image0'].shape[2]]],
                                         device=bbox0.device),
                    'bbox1': torch.tensor([[0.0, 0.0, data['image1'].shape[3],
                                           data['image1'].shape[2]]],
                                         device=bbox0.device),
                    'ratio0': torch.tensor([[1.0, 1.0]], device=bbox0.device),
                    'ratio1': torch.tensor([[1.0, 1.0]], device=bbox0.device),
                    'mask0': scaled_mask0,
                    'mask1': scaled_mask1,
                })
                
                if self.config['direct']:
                    matches = self.matcher(data)
                    pred.update(matches)
                    return pred
                
                # Extract features from full images
                if 'keypoints0' not in data:
                    pred0 = self.extractor({'image': data['image0']})
                    pred.update(dict((k + '0', v) for k, v in pred0.items()))
                if 'keypoints1' not in data:
                    pred1 = self.extractor({'image': data['image1']})
                    pred.update(dict((k + '1', v) for k, v in pred1.items()))

        else:
            # Standard processing without overlap estimation
            if not self.config['direct']:
                pred = extract_process(self.extractor, data)
                pred.update({
                    'bbox0': torch.tensor([[0.0, 0.0, data['image0'].shape[3],
                                           data['image0'].shape[2]]],
                                         device=data['image0'].device),
                    'bbox1': torch.tensor([[0.0, 0.0, data['image1'].shape[3],
                                           data['image1'].shape[2]]],
                                         device=data['image0'].device),
                    'ratio0': torch.tensor([[1.0, 1.0]], device=data['image0'].device),
                    'ratio1': torch.tensor([[1.0, 1.0]], device=data['image0'].device),
                    'mask0': torch.ones(1, 1, data['image0'].shape[2], data['image0'].shape[3]).to(bool).cuda(),
                    'mask1': torch.ones(1, 1, data['image1'].shape[2], data['image1'].shape[3]).to(bool).cuda(),
                })

        # Batch all features for consistent tensor operations
        data.update(pred)
        for k in data:
            if isinstance(data[k], (list, tuple)) and k not in [
                'overlap_scales0', 'overlap_scales1',
            ]:
                data[k] = torch.stack(data[k])

        # Perform matching
        matches = self.matcher(data)
        pred.update(matches)
        return pred


def save_h5(dict_to_save, filename):
    """Save dictionary to HDF5 file format."""
    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


def viz_pairs(output, image0, image1, name0, name1, mconf, kpts0, kpts1,
              mkpts0, mkpts1, bbox0=None, bbox1=None, mask0=None, mask1=None):
    """Visualize matching results between image pairs."""
    viz_path = output + '{}_{}_matches.png'.format(
        name0.split('/')[-1], name1.split('/')[-1])
    
    # Color code matches based on confidence threshold
    thres = 0.3
    r_a = (mconf < thres).astype(np.float32)
    g_a = (mconf > thres).astype(np.float32)
    b_a = np.zeros(mconf.shape)
    a_a = np.ones(mconf.shape)
    color = np.vstack((r_a, g_a, b_a, a_a)).transpose()

    text = [
        'Matcher',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]

    small_text = [
        'Image Pair: {}:{}'.format(name0.split('/')[-1], name1.split('/')[-1]),
    ]

    make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, viz_path,
        bbox0, bbox1, mask0, mask1, False, False, False, 'Matches', small_text,
    )


def main(
        config,
        input,
        input_pairs,
        output,
        with_desc=False,
        resize=None,
        resize_float=False,
        viz=False,
        save=False,
        evaluate=False,
        warp_origin=True,
):
    """Main pipeline for image matching evaluation.
    
    Args:
        config (dict): Configuration for extractor, matcher, and overlaper
        input (str): Input directory containing images
        input_pairs (str): Path to file listing image pairs
        output (str): Output directory for results
        with_desc (bool): Whether to save descriptors
        resize (list): Resize parameters for images
        resize_float (bool): Whether to resize after float conversion
        viz (bool): Whether to generate visualizations
        save (bool): Whether to save results to files
        evaluate (bool): Whether to perform online evaluation
        warp_origin (bool): Whether to warp keypoints to original scale
    """
    if resize is None:
        resize = [-1]
    
    # Initialize matching pipeline
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matching = Matching(config).eval().to(device)
    
    if not os.path.exists(output):
        os.makedirs(output)

    # Load image pairs
    with open(input_pairs, 'r') as f:
        pairs = [line.split() for line in f.readlines()]

    # Initialize result storage
    seq_keypoints = defaultdict(dict)
    seq_descriptors = defaultdict(dict)
    seq_matches = defaultdict(dict)
    seq_pose = defaultdict(dict)
    seq_inparams = defaultdict(dict)
    seq_scale = defaultdict(dict)
    seq_info = defaultdict(dict)
    
    # Process each image pair
    for i, pair in tqdm(enumerate(pairs), total=len(pairs)):
        name0, name1 = pair[:2]
        scale = float(pair[-1]) if pair.__len__() == 45 else -1

        # Extract processing parameters from config
        gray = config['extractor']['preprocessing']['grayscale']
        align = ''
        if 'disk' in config['extractor']['output']:
            align = 'disk'
        elif 'loftr' in config['matcher']['output']:
            align = 'loftr'
            with_desc = False
        
        # Determine scene name based on dataset type
        if 'megadepth' in input_pairs or 'MegaDepth' in input_pairs:
            scene = name0.split('/')[1]
        elif 'imc' in input_pairs:
            scene = name0.split('/')[0] + '/' + name0.split('/')[1]
        elif 'fuchi' in input_pairs:
            scene = name0.split('/')[-1][:-4]
        elif 'gl3d' in input_pairs:
            scene = name0.split('/')[-3]
        else:
            scene = name0.split('/')[0]

        # Choose processing pipeline based on configuration
        if config['overlaper'] is not None:
            results = preprocess_overlap_pipeline(
                input, name0, name1, device, resize, resize_float, gray,
                align, config, pair, matching, with_desc,
            )
        else:
            results = preprocess_match_pipeline(
                input, name0, name1, device, resize, resize_float, gray,
                align, config, pair, matching, with_desc,
            )
            
            # Handle ICP matcher special case
            if 'icp' in config['matcher']['model']['name']:
                viz_path = name0 + '_' + name1
                if viz and i % 10 == 0:
                    vis_aligned_image(results['mask0'], results['mask1'],
                                      results['T_0_1'], viz_path)
                seq_pose[scene]['{}-{}'.format(
                    name0.split('/')[-1][:-4],
                    name1.split('/')[-1][:-4])] = results['T_0_1']
                continue

        # Skip pairs with no valid matches
        if not results['mconf'].any():
            continue

        # Store keypoints and descriptors
        if 'loftr' in config['matcher']['output'] or config['overlaper'] is not None:
            im0, im1 = name0.split('/')[-1][:-4], name1.split('/')[-1][:-4]
            
            # Store keypoints for first image
            if '{}-{}'.format(im0, im1) not in seq_keypoints[scene]:
                seq_keypoints[scene]['{}-{}'.format(im0, im1)] = results['kpts0']
                if config['overlaper'] is not None and not warp_origin:
                    seq_inparams[scene]['{}-{}'.format(im0, im1)] = np.concatenate(
                        (np.array(results['scales0']), results['oxy0'], results['ratio0']),
                        axis=-1,
                    )
                if with_desc:
                    seq_descriptors[scene]['{}-{}'.format(im0, im1)] = results['desc0']
            
            # Store keypoints for second image
            if '{}-{}'.format(im1, im0) not in seq_keypoints[scene]:
                seq_keypoints[scene]['{}-{}'.format(im1, im0)] = results['kpts1']
                if config['overlaper'] is not None and not warp_origin:
                    seq_inparams[scene]['{}-{}'.format(im1, im0)] = np.concatenate(
                        (np.array(results['scales1']), results['oxy1'], results['ratio1']),
                        axis=-1,
                    )
                if with_desc:
                    seq_descriptors[scene]['{}-{}'.format(im1, im0)] = results['desc1']
        else:
            # Store per-image keypoints for non-LoFTR matchers
            if name0 not in seq_keypoints[scene]:
                seq_keypoints[scene][name0.split('/')[-1][:-4]] = results['kpts0']
                if with_desc:
                    seq_descriptors[scene][name0.split('/')[-1][:-4]] = results['desc0']

            if name1 not in seq_keypoints[scene]:
                seq_keypoints[scene][name1.split('/')[-1][:-4]] = results['kpts1']
                if with_desc:
                    seq_descriptors[scene][name1.split('/')[-1][:-4]] = results['desc1']
        
        # Store match indices
        seq_matches[scene]['{}-{}'.format(
            name0.split('/')[-1][:-4], name1.split('/')[-1][:-4]
        )] = np.concatenate([[results['index0']], [results['index1']]])
        
        # Store additional information including timing data
        overlap_time = results.get('overlap_time', 0)
        if isinstance(overlap_time, torch.Tensor):
            overlap_time = overlap_time.item()
        
        seq_info[scene].update({
            '{}-{}'.format(name0.split('/')[-1][:-4], name1.split('/')[-1][:-4]): {
                'K0': results['K0'],
                'K1': results['K1'],
                'T_0to1': results['T_0to1'],
                'mkpts0': results['mkpts0'],
                'mkpts1': results['mkpts1'],
                'conf': np.mean(results['conf']),
                'scale': scale,
                'overlap_time': overlap_time
            }
        })
        
        # Store scale ratios if available
        if 'ratio0' in results and 'ratio1' in results:
            seq_scale[scene]['{}-{}'.format(
                name0.split('/')[-1][:-4], name1.split('/')[-1][:-4]
            )] = min(
                results['ratio0'][0] / results['ratio1'][0],
                results['ratio1'][0] / results['ratio0'][0],
            )

        # Generate visualizations
        if viz and i % 10 == 0:
            if not os.path.exists(os.path.join(output, 'viz')):
                os.makedirs(os.path.join(output, 'viz'))

            viz_pairs(
                os.path.join(output, 'viz/{}_'.format(scene.replace('/', '-'))),
                results['image0'], results['image1'], name0, name1,
                results['mconf'], results['kpts0_ori'], results['kpts1_ori'],
                results['kpts0_ori'][results['index0']],
                results['kpts1_ori'][results['index1']],
                results['bbox0'], results['bbox1'],
                results['mask0'], results['mask1'],
            )
    
    # Save results to HDF5 files
    if save:
        for k in seq_keypoints.keys():
            if not os.path.exists(os.path.join(output, k)):
                os.makedirs(os.path.join(output, k))
            
            if with_desc:
                save_h5(seq_descriptors[k], os.path.join(output, k, 'descriptors.h5'))
            save_h5(seq_keypoints[k], os.path.join(output, k, 'keypoints.h5'))
            save_h5(seq_matches[k], os.path.join(output, k, 'matches.h5'))
            np.savez(os.path.join(output, k, 'infos.npz'), **seq_info[k])
            
            if len(seq_inparams) > 0:
                save_h5(seq_inparams[k], os.path.join(output, k, 'inparams.h5'))
            if len(seq_scale) > 0:
                save_h5(seq_scale[k], os.path.join(output, k, 'scales.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--input_pairs', type=str, default='assets/pairs.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--input_dir', type=str, default='assets/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory for results and visualizations',
    )
    parser.add_argument(
        '--matcher', choices={
            'superglue_outdoor', 'superglue_disk', 'superglue_swin_disk',
            'superglue_indoor', 'NN', 'disk', 'cotr', 'loftr',
        }, default='indoor', help='Matcher type',
    )
    parser.add_argument(
        '--extractor', choices={
            'superpoint_aachen', 'superpoint_inloc', 'd2net-ss', 'r2d2-desc',
            'context-desc', 'landmark', 'aslfeat-desc', 'disk-desc', 'swin-disk-desc',
        }, default='superpoint_aachen', help='Feature extractor type',
    )
    parser.add_argument(
        '--overlaper', choices={'oetr', 'oetr_imc', 'detmatcher', 'samatcher'},
        default=None, help='Overlap estimator type'
    )
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[-1],
        help='Resize parameters: two numbers for exact dimensions, '
             'one number for max dimension, -1 for no resize',
    )
    parser.add_argument(
        '--with_desc', action='store_true', help='Save descriptors'
    )
    parser.add_argument(
        '--viz', action='store_true', help='Generate visualization'
    )
    parser.add_argument(
        '--landmark', action='store_true', help='Use landmark keypoints'
    )
    parser.add_argument(
        '--direct', action='store_true',
        help='Direct matching without keypoint extraction'
    )
    parser.add_argument(
        '--save', action='store_true', help='Save match results'
    )
    parser.add_argument(
        '--evaluate', action='store_true', help='Perform online evaluation'
    )
    parser.add_argument(
        '--warp_origin', action='store_false',
        help='Warp keypoints to original image scale'
    )
    
    opt = parser.parse_args()
    
    # Build configuration from command line arguments
    extractor_conf = extract_features.confs[opt.extractor]
    matcher_conf = match_features.confs[opt.matcher]
    overlaper_conf = overlap_features.confs[opt.overlaper] if opt.overlaper else None
    
    config = {
        'landmark': opt.landmark,
        'extractor': extractor_conf,
        'matcher': matcher_conf,
        'direct': opt.direct,
        'overlaper': overlaper_conf,
    }
    
    # Generate output path
    if overlaper_conf is not None:
        output_path = os.path.join(
            opt.output_dir,
            opt.extractor + '_' + opt.matcher + '_' + opt.overlaper + '/',
        )
    else:
        output_path = os.path.join(opt.output_dir, opt.extractor + '_' + opt.matcher + '/')
    
    # Run main pipeline
    main(
        config, opt.input_dir, opt.input_pairs, output_path, opt.with_desc,
        opt.resize, viz=opt.viz, save=opt.save, evaluate=opt.evaluate,
        warp_origin=opt.warp_origin,
    )