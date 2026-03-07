import os
import h5py
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from collections import defaultdict

from dloc.api import extract_process
from dloc.core.utils.base_model import dynamic_load
from dloc.core.match_features import preprocess_match_pipeline
from dloc.core.overlap_features import preprocess_overlap_pipeline
from dloc.core.utils.utils import make_matching_plot, tensor_overlap_crop, vis_aligned_image, visualize_box_mask_constraint_pair
from dloc.core import extract_features, extractors, match_features, matchers, overlap_features, overlaps

torch.set_grad_enabled(False)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

t_thres=0.3

# --- New: gridify binary masks for visualization (does NOT affect matching) ---
def _gridify_bin_mask(mask, win: int = 16, thresh: float = 0.5):
    """
    Gridify a (binary-ish) mask by NxN windows:
      if any pixel in a window is valid -> mark the whole window valid.
    Supports torch.Tensor OR numpy array.
    Returns same "kind" as input (torch -> torch, numpy -> numpy).
    """
    if mask is None:
        return None

    # --- torch path ---
    if isinstance(mask, torch.Tensor):
        m = mask
        if m.ndim == 2:
            m = m[None, None]
        elif m.ndim == 3:
            m = m[None]
        elif m.ndim != 4:
            return mask
        m = (m > float(thresh)).float()
        H, W = int(m.shape[-2]), int(m.shape[-1])
        if win <= 1:
            return m[..., :H, :W]
        pad_h = (win - (H % win)) % win
        pad_w = (win - (W % win)) % win
        if pad_h or pad_w:
            m = F.pad(m, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        pooled = F.max_pool2d(m, kernel_size=win, stride=win)
        up = pooled.repeat_interleave(win, dim=-2).repeat_interleave(win, dim=-1)
        return up[..., :H, :W]

    # --- numpy fallback (keeps minimal + robust) ---
    m = np.asarray(mask)
    m = np.squeeze(m).astype(np.float32)
    if m.ndim != 2:
        return mask
    m = (m > float(thresh)).astype(np.float32)
    H, W = m.shape
    if win <= 1:
        return m
    pad_h = (win - (H % win)) % win
    pad_w = (win - (W % win)) % win
    if pad_h or pad_w:
        m = np.pad(m, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
    H2, W2 = m.shape
    bh, bw = H2 // win, W2 // win
    blocks = m.reshape(bh, win, bw, win)
    pooled = blocks.max(axis=(1, 3))  # (bh, bw)
    up = np.repeat(np.repeat(pooled, win, axis=0), win, axis=1)
    return up[:H, :W].astype(np.float32)

# --- New: make sure viz has bin/crop masks even if pipeline didn't return them ---
def _ensure_viz_masks(results, bin_thresh=0.5, extractor_name=''):
    # mask*_bin: if missing, derive from mask* (soft) for visualization purposes
    if results.get('mask0_bin', None) is None and results.get('mask0', None) is not None:
        m0 = results['mask0']
        results['mask0_bin'] = (m0 > bin_thresh).float() if isinstance(m0, torch.Tensor) else (np.asarray(m0) > bin_thresh).astype(np.float32)
    if results.get('mask1_bin', None) is None and results.get('mask1', None) is not None:
        m1 = results['mask1']
        results['mask1_bin'] = (m1 > bin_thresh).float() if isinstance(m1, torch.Tensor) else (np.asarray(m1) > bin_thresh).astype(np.float32)

    # crop_mask*: if missing, try to reconstruct (only for visualization tint inside bbox)
    if results.get('crop_mask0', None) is None and results.get('crop_mask1', None) is None:
        try:
            if all(k in results for k in ('mask0_bin', 'mask1_bin', 'bbox0', 'bbox1')):
                m0 = results['mask0_bin']
                m1 = results['mask1_bin']
                b0 = results['bbox0']
                b1 = results['bbox1']
                if not isinstance(m0, torch.Tensor):
                    m0 = torch.from_numpy(np.asarray(m0)).float()
                if not isinstance(m1, torch.Tensor):
                    m1 = torch.from_numpy(np.asarray(m1)).float()
                if m0.ndim == 2: m0 = m0[None, None]
                if m1.ndim == 2: m1 = m1[None, None]
                if not isinstance(b0, torch.Tensor):
                    b0 = torch.from_numpy(np.asarray(b0)).float()
                if not isinstance(b1, torch.Tensor):
                    b1 = torch.from_numpy(np.asarray(b1)).float()
                b0 = b0.reshape(1, -1)[:, :4]
                b1 = b1.reshape(1, -1)[:, :4]
                # same call style as Matching.forward (viz only)
                cm0, cm1, _, _ = tensor_overlap_crop(m0, b0, m1, b1, extractor_name, size_divisor=1)
                results['crop_mask0'] = cm0
                results['crop_mask1'] = cm1
        except Exception:
            pass
    return results

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

    # --- New helper: ensure consistent default fields in pred ---
    def _ensure_pred_defaults(self, pred, data):
        """
        Ensure pred dict contains bbox0/bbox1, ratio0/ratio1, mask0/mask1,
        mask0_bin/mask1_bin and mask_prefiltered. Uses image sizes/devices from data.
        """
        try:
            # attempt to infer device and image shapes from data
            device = None
            if isinstance(data, dict):
                if 'image0' in data and hasattr(data['image0'], 'device'):
                    device = data['image0'].device
                    H0 = int(data['image0'].shape[2])
                    W0 = int(data['image0'].shape[3])
                if 'image1' in data and device is None and hasattr(data['image1'], 'device'):
                    device = data['image1'].device
            if device is None:
                device = torch.device('cpu')

            # bbox defaults: [xmin, ymin, xmax, ymax] full-image
            if 'bbox0' not in pred or pred.get('bbox0') is None:
                pred['bbox0'] = torch.tensor([[0.0, 0.0, float(W0), float(H0)]],
                                             device=device)
            if 'bbox1' not in pred or pred.get('bbox1') is None:
                # if image1 available, derive sizes, else reuse image0 dims
                if 'image1' in data:
                    H1 = int(data['image1'].shape[2]); W1 = int(data['image1'].shape[3])
                else:
                    H1, W1 = H0, W0
                pred['bbox1'] = torch.tensor([[0.0, 0.0, float(W1), float(H1)]],
                                             device=device)

            # ratio defaults
            if 'ratio0' not in pred:
                pred['ratio0'] = torch.tensor([[1.0, 1.0]], device=device)
            if 'ratio1' not in pred:
                pred['ratio1'] = torch.tensor([[1.0, 1.0]], device=device)

            # Only create mask-related defaults if an overlaper is configured.
            if self.config.get('overlaper') is not None:
                # soft masks default (float in [0,1]), plus binary masks
                if 'mask0' not in pred or pred.get('mask0') is None:
                    mask0 = torch.ones(1, 1, H0, W0, dtype=torch.float32, device=device)
                    pred['mask0'] = mask0
                else:
                    # ensure float tensor on correct device
                    m0 = pred['mask0']
                    if isinstance(m0, torch.Tensor):
                        pred['mask0'] = m0.detach().to(device).float()
                if 'mask1' not in pred or pred.get('mask1') is None:
                    mask1 = torch.ones(1, 1, H1, W1, dtype=torch.float32, device=device)
                    pred['mask1'] = mask1
                else:
                    m1 = pred['mask1']
                    if isinstance(m1, torch.Tensor):
                        pred['mask1'] = m1.detach().to(device).float()

                # binary masks (use threshold 0.5 on soft mask if not present)
                if 'mask0_bin' not in pred or pred.get('mask0_bin') is None:
                    pred['mask0_bin'] = (pred['mask0'] > 0.5).float()
                if 'mask1_bin' not in pred or pred.get('mask1_bin') is None:
                    pred['mask1_bin'] = (pred['mask1'] > 0.5).float()

                # ensure mask_prefiltered exists (False by default)
                if 'mask_prefiltered' not in pred:
                    pred['mask_prefiltered'] = torch.tensor([False], device=device)
            else:
                # If no overlaper configured, avoid injecting masks into pred.
                # Still keep mask_prefiltered default for downstream checks (as boolean flag only),
                # but as a plain Python False (not a tensor) to avoid adding mask tensors.
                pred.setdefault('mask_prefiltered', False)
        except Exception:
            # best-effort: do not crash the pipeline on helper errors
            pass

    # --- New: pad-to-divisor helper for feature inputs (right/bottom zero padding) ---
    def _pad_to_divisor(self, img: torch.Tensor, divisor: int) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            return img
        if img.ndim != 4:
            return img
        H, W = int(img.shape[2]), int(img.shape[3])
        pad_h = (divisor - (H % divisor)) % divisor
        pad_w = (divisor - (W % divisor)) % divisor
        if pad_h == 0 and pad_w == 0:
            return img
        return F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

    def forward(self, data, with_overlap=False):
        """Run extractors and matchers on image pair.
        
        Args:
            data (dict): Input data with keys ['image0', 'image1'] and optional overlap data
            with_overlap (bool): Whether to use overlap estimation
            
        Returns:
            dict: Matching results including keypoints, matches, and scores
        """
        # Ensure DISK extractor inputs meet U-Net size constraints (H,W multiples of 16)
        if 'disk' in (self.extractor_name or ''):
            # pad only the original images used by extractors/matchers
            for k in ('image0', 'image1'):
                if k in data and isinstance(data[k], torch.Tensor):
                    data[k] = self._pad_to_divisor(data[k], 16)

        # Skip extractor for direct matching without overlap
        if self.config['direct'] and not with_overlap:
            # matcher may return dict missing mask/bbox defaults -> ensure them
            matches = self.matcher(data)
            try:
                self._ensure_pred_defaults(matches, data)
            except Exception:
                pass
            return matches

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
            bbox0, bbox1, mask0, mask1, mask0_o, mask1_o = self.overlap({
                'image0': data['overlap_image0'],
                'image1': data['overlap_image1']
            })
            overlap_time = (time.time() - overlap_start_time) * 1000
            pred['overlap_time'] = torch.tensor([overlap_time], device=self.device)

            # Original image dimensions
            H0, W0 = int(data['image0'].shape[2]), int(data['image0'].shape[3])
            H1, W1 = int(data['image1'].shape[2]), int(data['image1'].shape[3])

            # Recover "content area size" (unpadded) of overlaper input from overlap_scales (based on content size)
            sx0 = float(data['overlap_scales0'][0].item())
            sy0 = float(data['overlap_scales0'][1].item())
            sx1 = float(data['overlap_scales1'][0].item())
            sy1 = float(data['overlap_scales1'][1].item())
            content_w0 = max(1, int(round(W0 / sx0)))
            content_h0 = max(1, int(round(H0 / sy0)))
            content_w1 = max(1, int(round(W1 / sx1)))
            content_h1 = max(1, int(round(H1 / sy1)))

            # Clamp bbox within "content area" in overlaper coordinate system (remove right/bottom padding)
            bbox0[:, 0::2] = bbox0[:, 0::2].clamp(0, content_w0 - 1)
            bbox0[:, 1::2] = bbox0[:, 1::2].clamp(0, content_h0 - 1)
            bbox1[:, 0::2] = bbox1[:, 0::2].clamp(0, content_w1 - 1)
            bbox1[:, 1::2] = bbox1[:, 1::2].clamp(0, content_h1 - 1)

            # Scale bbox from overlaper content coordinates back to original image coordinates ([sx,sy,sx,sy])
            bbox0 = bbox0 * data['overlap_scales0']
            bbox1 = bbox1 * data['overlap_scales1']

            # Finally ensure bbox is within the original image range
            bbox0[:, 0::2] = bbox0[:, 0::2].clamp(0, W0 - 1)
            bbox0[:, 1::2] = bbox0[:, 1::2].clamp(0, H0 - 1)
            bbox1[:, 0::2] = bbox1[:, 0::2].clamp(0, W1 - 1)
            bbox1[:, 1::2] = bbox1[:, 1::2].clamp(0, H1 - 1)

            # Crop mask to "content area" then resize to original image to avoid coordinate shift caused by padding stretching
            mask0_c = mask0[:content_h0, :content_w0]
            mask1_c = mask1[:content_h1, :content_w1]

            raw_mask0 = F.interpolate(mask0_c.unsqueeze(0).unsqueeze(0),
                                      size=(H0, W0), mode='bilinear', align_corners=False)
            raw_mask1 = F.interpolate(mask1_c.unsqueeze(0).unsqueeze(0),
                                      size=(H1, W1), mode='bilinear', align_corners=False)
            soft_mask0 = raw_mask0.sigmoid()
            soft_mask1 = raw_mask1.sigmoid()
            threshold = t_thres
            bin_mask0 = (soft_mask0 > threshold).float()
            bin_mask1 = (soft_mask1 > threshold).float()

            mask0_o = mask0_o.sigmoid()
            mask1_o = mask1_o.sigmoid()

            # --- Fix: compute soft-box directly from soft_mask (global), no base bbox offset ---
            def _bbox_from_soft_mask_global(soft_mask_t, th=0.05):
                # soft_mask_t: tensor [1,1,H,W] or [H,W] at original image resolution
                if soft_mask_t is None:
                    return None
                m = soft_mask_t.detach().cpu().float().squeeze().numpy()
                if m.ndim != 2:
                    return None
                mask_bin = (m > th).astype(np.uint8)
                ys, xs = np.where(mask_bin)
                if xs.size == 0 or ys.size == 0:
                    return None
                xmin, xmax = int(xs.min()), int(xs.max())
                ymin, ymax = int(ys.min()), int(ys.max())
                # ensure a valid box (at least 1px wide/high)
                if xmax <= xmin: xmax = xmin + 1
                if ymax <= ymin: ymax = ymin + 1
                return (xmin, ymin, xmax, ymax)

            mb0 = _bbox_from_soft_mask_global(soft_mask0, th=t_thres)
            mb1 = _bbox_from_soft_mask_global(soft_mask1, th=t_thres)

            mode = 'intersect' # 'intersect' or 'union'

            def _apply_box_combination(bbox_t, mb_tuple, comb_mode='union'):
                """Combine existing bbox_t (torch [1,4]) with mb_tuple (xmin,ymin,xmax,ymax)
                   comb_mode: 'union' or 'intersect'. If intersection is empty, keep original bbox_t.
                """
                if mb_tuple is None:
                    return bbox_t
                try:
                    b = bbox_t.detach().cpu().numpy().reshape(-1)[:4].astype(float)
                except Exception:
                    return bbox_t
                bx0, by0, bx1, by1 = b
                mx0, my0, mx1, my1 = [float(x) for x in mb_tuple]
                if comb_mode == 'union':
                    nx0 = min(bx0, mx0)
                    ny0 = min(by0, my0)
                    nx1 = max(bx1, mx1)
                    ny1 = max(by1, my1)
                elif comb_mode == 'intersect':
                    nx0 = max(bx0, mx0)
                    ny0 = max(by0, my0)
                    nx1 = min(bx1, mx1)
                    ny1 = min(by1, my1)
                    # If intersection is empty, keep original bbox_t to avoid invalid box
                    if nx1 <= nx0 or ny1 <= ny0:
                        return bbox_t
                else:
                    # Unknown mode -> no change
                    return bbox_t
                return torch.tensor([[float(nx0), float(ny0), float(nx1), float(ny1)]],
                                    device=bbox_t.device, dtype=bbox_t.dtype)

            # Apply combination strategy (when mb exists)
            try:
                if mb0 is not None:
                    bbox0 = _apply_box_combination(bbox0, mb0, mode)
                if mb1 is not None:
                    bbox1 = _apply_box_combination(bbox1, mb1, mode)
            except Exception:
                # Safe fallback: keep original bbox on error
                pass
            # --- end fix ---

            # Calculate overlap region dimensions
            # (convert to Python ints to avoid ambiguous tensor comparisons)
            bw0, bh0 = (
                int((bbox0[0][2] - bbox0[0][0]).item()),
                int((bbox0[0][3] - bbox0[0][1]).item()),
            )
            bw1, bh1 = (
                int((bbox1[0][2] - bbox1[0][0]).item()),
                int((bbox1[0][3] - bbox1[0][1]).item()),
            )

            # --- New: fallback to full-image when mask or bbox are empty/too small ---
            def _mask_or_box_bad(bbox_t, soft_mask_t, min_side=8, min_mask_sum=1e-3):
                try:
                    bw = float((bbox_t[0][2] - bbox_t[0][0]).item())
                    bh = float((bbox_t[0][3] - bbox_t[0][1]).item())
                    if bw < min_side or bh < min_side:
                        return True
                except Exception:
                    return True
                if soft_mask_t is None:
                    return True
                try:
                    s = float(soft_mask_t.detach().cpu().float().sum())
                    # also check max value so a tiny nonzero sum (noise) does not pass
                    mmax = float(soft_mask_t.detach().cpu().float().max())
                    if s < min_mask_sum or mmax <= 1e-4:
                        return True
                except Exception:
                    return True
                return False

            use_full_image = False
            try:
                if _mask_or_box_bad(bbox0, soft_mask0) or _mask_or_box_bad(bbox1, soft_mask1):
                    use_full_image = True
            except Exception:
                use_full_image = False

            # Calculate overlap scale ratio safely
            overlap_scores = max(
                bw0 / max(bw1, 1e-6),
                bh0 / max(bh1, 1e-6),
                bw1 / max(bw0, 1e-6),
                bh1 / max(bh0, 1e-6),
            )

            # Process overlap regions if they meet size and ratio requirements
            if (not use_full_image) and min(bw0, bh0, bw1, bh1) > 1 and (
                    (data['dataset_name'] == 'pragueparks-val'
                     and overlap_scores > 2.0)
                    or data['dataset_name'] != 'pragueparks-val'):
                
                # --- Changed: choose size_divisor satisfying matcher/extractor requirements ---
                size_div = 1
                if self.matcher_name == 'loftr':
                    size_div = 8
                # DISK extractor requires multiples of 16
                if 'disk' in (self.extractor_name or ''):
                    size_div = max(size_div, 16)
                self.size_divisor = size_div

                if self.config['overlaper']['model']['name'] == 'samatcher':
                    # use binary masks for crop-mask generation
                    crop_mask0, crop_mask1, _, _ = tensor_overlap_crop(
                        bin_mask0, bbox0, bin_mask1, bbox1,
                        self.extractor_name, self.size_divisor,
                    )
                
                # Crop overlap regions with enforced size_divisor
                overlap0, overlap1, ratio0, ratio1 = tensor_overlap_crop(
                    data['image0'], bbox0, data['image1'], bbox1,
                    self.extractor_name, self.size_divisor,
                )

                # store soft masks for downstream reweight (continuous values in [0,1])
                pred.update({
                    'bbox0': bbox0,
                    'bbox1': bbox1,
                    'ratio0': torch.tensor(ratio0, device=bbox0.device),
                    'ratio1': torch.tensor(ratio1, device=bbox0.device),
                    'mask0': soft_mask0,
                    'mask1': soft_mask1,
                    'mask0_o': mask0_o,
                    'mask1_o': mask1_o,
                    # keep binary masks handy for any components that need them
                    'mask0_bin': bin_mask0,
                    'mask1_bin': bin_mask1,
                })
                # Save crop masks (from tensor_overlap_crop) for visualization / debugging.
                # crop_mask0/crop_mask1 exist when overlaper model == 'samatcher'
                try:
                    pred['crop_mask0'] = crop_mask0
                    pred['crop_mask1'] = crop_mask1
                except NameError:
                    # crop masks not available in this branch - ignore
                    pass

                # Direct matching on overlap regions
                if self.config['direct']:
                    matches = self.matcher({'image0': overlap0, 'image1': overlap1})
                    pred.update(matches)
                    return pred
                
                # Extract features from overlap regions
                if 'keypoints0' not in data:
                    pred0 = self.extractor({'image': overlap0})
                    if self.config['overlaper']['model']['name'] == 'samatcher':
                        # --- New: vectorized & soft prefilter (optional, low-thresh) ---
                        pre_th = t_thres  # low threshold for prefilter (or set to 0.0 to disable)
                        dilate_r = 3   # small tolerance radius
                        m0 = crop_mask0.float().squeeze()  # [H,W], values in {0,1} or [0,1]
                        k0 = pred0['keypoints'][0]          # [N,2] in overlap coords (x,y)
                        H0, W0 = int(m0.shape[-2]), int(m0.shape[-1])
                        # round-to-nearest index with clamping
                        xi = k0[:, 0].round().long().clamp(0, W0 - 1)
                        yi = k0[:, 1].round().long().clamp(0, H0 - 1)
                        # local max pooling by integer neighborhood (fast and simple)
                        valid0 = []
                        with torch.no_grad():
                            for x, y in zip(xi.tolist(), yi.tolist()):
                                x0, x1 = max(0, x - dilate_r), min(W0, x + dilate_r + 1)
                                y0, y1 = max(0, y - dilate_r), min(H0, y + dilate_r + 1)
                                valid0.append(float(m0[y0:y1, x0:x1].max()) >= pre_th)
                        valid0 = torch.tensor(valid0, device=k0.device, dtype=torch.bool)
                        # --- New: record soft-prefilter stats for debug viz ---
                        n_raw0 = int(k0.shape[0])
                        n_keep0 = int(valid0.sum().item())
                        pred0['prefilter_raw'] = [torch.tensor(n_raw0, device=k0.device)]
                        pred0['prefilter_kept'] = [torch.tensor(n_keep0, device=k0.device)]
                        pred0['prefilter_valid_mask'] = [valid0]
                        if n_keep0 >= 3:
                            pred0['keypoints'] = [pred0['keypoints'][0][valid0]]
                            pred0['scores'] = [pred0['scores'][0][valid0]]
                            pred0['descriptors'] = [pred0['descriptors'][0][:, valid0]]
                            # mark as prefiltered to avoid double hard filtering later
                            pred0['mask_prefiltered'] = [torch.tensor(True, device=k0.device)]
                    pred.update(dict((k + '0', v) for k, v in pred0.items()))
                if 'keypoints1' not in data:
                    pred1 = self.extractor({'image': overlap1})
                    if self.config['overlaper']['model']['name'] == 'samatcher':
                        pre_th = 0.10
                        dilate_r = 1
                        m1 = crop_mask1.float().squeeze()
                        k1 = pred1['keypoints'][0]
                        H1, W1 = int(m1.shape[-2]), int(m1.shape[-1])
                        xi = k1[:, 0].round().long().clamp(0, W1 - 1)
                        yi = k1[:, 1].round().long().clamp(0, H1 - 1)
                        valid1 = []
                        with torch.no_grad():
                            for x, y in zip(xi.tolist(), yi.tolist()):
                                x0, x1 = max(0, x - dilate_r), min(W1, x + dilate_r + 1)
                                y0, y1 = max(0, y - dilate_r), min(H1, y + dilate_r + 1)
                                valid1.append(float(m1[y0:y1, x0:x1].max()) >= pre_th)
                        valid1 = torch.tensor(valid1, device=k1.device, dtype=torch.bool)
                        # --- New: record soft-prefilter stats for debug viz ---
                        n_raw1 = int(k1.shape[0])
                        n_keep1 = int(valid1.sum().item())
                        pred1['prefilter_raw'] = [torch.tensor(n_raw1, device=k1.device)]
                        pred1['prefilter_kept'] = [torch.tensor(n_keep1, device=k1.device)]
                        pred1['prefilter_valid_mask'] = [valid1]
                        if n_keep1 >= 3:
                            pred1['keypoints'] = [pred1['keypoints'][0][valid1]]
                            pred1['scores'] = [pred1['scores'][0][valid1]]
                            pred1['descriptors'] = [pred1['descriptors'][0][:, valid1]]
                            pred1['mask_prefiltered'] = [torch.tensor(True, device=k1.device)]
                    pred.update(dict((k + '1', v) for k, v in pred1.items()))

            else:
                # Use full images when overlap regions are too small OR predicted masks/boxes were bad
                # (this ensures we never pass None/very small crops/masks to extractor/matcher)
                pred.update({
                    'bbox0': torch.tensor([[0.0, 0.0, data['image0'].shape[3],
                                           data['image0'].shape[2]]],
                                         device=bbox0.device),
                    'bbox1': torch.tensor([[0.0, 0.0, data['image1'].shape[3],
                                           data['image1'].shape[2]]],
                                         device=bbox0.device),
                    'ratio0': torch.tensor([[1.0, 1.0]], device=bbox0.device),
                    'ratio1': torch.tensor([[1.0, 1.0]], device=bbox0.device),
                    # if masks were unreliable, replace with safe all-ones masks
                    'mask0': soft_mask0 if not use_full_image else torch.ones_like(soft_mask0),
                    'mask1': soft_mask1 if not use_full_image else torch.ones_like(soft_mask1),
                    'mask0_o': mask0_o if not use_full_image else torch.ones_like(mask0_o),
                    'mask1_o': mask1_o if not use_full_image else torch.ones_like(mask1_o),
                    'mask0_bin': bin_mask0 if not use_full_image else torch.ones_like(bin_mask0),
                    'mask1_bin': bin_mask1 if not use_full_image else torch.ones_like(bin_mask1),
                })
                
                if self.config['direct']:
                    matches = self.matcher(data)
                    pred.update(matches)
                    return pred
                
                # --- Changed: pad full-image inputs for DISK extractor to multiples of 16 ---
                if 'keypoints0' not in data:
                    img0_feat = data['image0']
                    if 'disk' in (self.extractor_name or ''):
                        img0_feat = self._pad_to_divisor(img0_feat, 16)
                    pred0 = self.extractor({'image': img0_feat})
                    pred.update(dict((k + '0', v) for k, v in pred0.items()))
                if 'keypoints1' not in data:
                    img1_feat = data['image1']
                    if 'disk' in (self.extractor_name or ''):
                        img1_feat = self._pad_to_divisor(img1_feat, 16)
                    pred1 = self.extractor({'image': img1_feat})
                    pred.update(dict((k + '1', v) for k, v in pred1.items()))
        else:
            # Standard processing without overlap estimation
            pred = extract_process(self.extractor, data)
            # device-aware default bbox/mask (soft mask float + binary mask)
            device = data['image0'].device
            H0 = int(data['image0'].shape[2]); W0 = int(data['image0'].shape[3])
            H1 = int(data['image1'].shape[2]); W1 = int(data['image1'].shape[3])

            # Always set bbox/ratio defaults
            defaults = {
                'bbox0': torch.tensor([[0.0, 0.0, float(W0), float(H0)]], device=device),
                'bbox1': torch.tensor([[0.0, 0.0, float(W1), float(H1)]], device=device),
                'ratio0': torch.tensor([[1.0, 1.0]], device=device),
                'ratio1': torch.tensor([[1.0, 1.0]], device=device),
            }

            # Only add mask tensors when an overlaper exists (otherwise avoid mask fields)
            if self.config.get('overlaper') is not None:
                defaults.update({
                    # soft masks (float in [0,1]) and binary masks kept separately
                    'mask0': torch.ones(1, 1, H0, W0, dtype=torch.float32, device=device),
                    'mask1': torch.ones(1, 1, H1, W1, dtype=torch.float32, device=device),
                    'mask0_o': torch.ones(1, 1, 256, 256, dtype=torch.float32, device=device),
                    'mask1_o': torch.ones(1, 1, 256, 256, dtype=torch.float32, device=device),
                    'mask0_bin': torch.ones(1, 1, H0, W0, dtype=torch.float32, device=device),
                    'mask1_bin': torch.ones(1, 1, H1, W1, dtype=torch.float32, device=device),
                })
                # also ensure prefiler flag present as tensor when overlaper exists
                defaults.setdefault('mask_prefiltered', torch.tensor([False], device=device))
            else:
                # If no overlaper, keep mask_prefiltered as plain False for checks (no mask tensors)
                defaults.setdefault('mask_prefiltered', False)

            pred.update(defaults)
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

        # --- New: propagate prefilter flag for mask_filter auto-mode ---
        try:
            pref0 = pred.get('mask_prefiltered0', None)
            pref1 = pred.get('mask_prefiltered1', None)
            def _to_bool(x):
                if x is None: return False
                if isinstance(x, (list, tuple)): x = x[0]
                if isinstance(x, torch.Tensor): return bool(x.detach().cpu().item())
                return bool(x)
            if _to_bool(pref0) or _to_bool(pref1):
                pred['mask_prefiltered'] = torch.tensor([True], device=self.device)
        except Exception:
            pass

        return pred


def save_h5(dict_to_save, filename):
    """Save dictionary to HDF5 file format."""
    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:
            f.create_dataset(key, data=dict_to_save[key])


def viz_pairs(output, image0, image1, name0, name1, mconf, kpts0, kpts1,
              mkpts0, mkpts1, bbox0=None, bbox1=None, mask0=None, mask1=None,
              crop_mask0=None, crop_mask1=None, match_keep=None, mask0_o=None, mask1_o=None):
    """Visualize matching results between image pairs."""
    viz_path = output + '{}_{}_matches.png'.format(
        name0.split('/')[-1], name1.split('/')[-1])

    # make sure match arrays are numpy arrays and consistent lengths
    try:
        if mconf is None:
            mconf = np.array([])
        else:
            mconf = np.asarray(mconf)
            # replace NaNs and infs
            mconf = np.nan_to_num(mconf, nan=0.0, posinf=1.0, neginf=0.0)

        # color code matches based on confidence threshold (safe for empty)
        thres = 0.4
        if mconf.size == 0:
            color = np.zeros((0, 4), dtype=np.float32)
        else:
            r_a = (mconf < thres).astype(np.float32)
            g_a = (mconf >= thres).astype(np.float32)
            b_a = np.zeros(mconf.shape, dtype=np.float32)
            a_a = np.ones(mconf.shape, dtype=np.float32)
            color = np.vstack((r_a, g_a, b_a, a_a)).transpose()
    except Exception as e:
        print(f"viz_pairs: failed processing confidences for {name0}, {name1}: {e}")
        color = np.zeros((0, 4), dtype=np.float32)

    text = [
        'Matcher',
        'Keypoints: {}:{}'.format(0 if kpts0 is None else len(kpts0),
                                  0 if kpts1 is None else len(kpts1)),
        'Matches: {}'.format(0 if mkpts0 is None else len(mkpts0)),
    ]

    small_text = [
        'Image Pair: {}:{}'.format(name0.split('/')[-1], name1.split('/')[-1]),
    ]
    # Add prefilter / postfilter stats when available
    try:
        if crop_mask0 is not None or crop_mask1 is not None:
            small_text.append('Has crop_mask')
        if match_keep is not None:
            mk = np.asarray(match_keep)
            small_text.append('Postfilter kept: {}/{}'.format(int(mk.sum()), len(mk)))
    except Exception:
        pass

    # normalize bbox to (4,) int32
    def _norm_bbox(b):
        if b is None:
            return None
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()
        b = np.array(b).reshape(-1)[:4].astype(np.int32)
        return b

    b0 = _norm_bbox(bbox0)
    b1 = _norm_bbox(bbox1)

    # derive labels to show on each image (path starting from scene)
    # label0 = name0  # name0 already contains scene/...; adjust if you want a shorter form
    # label1 = name1

    # Avoid crashing the pipeline on plotting errors
    try:
        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, viz_path,
            b0, b1, mask0, mask1, False, False, False, 'Matches', small_text,
            # label0=label0, label1=label1,
            label0=None, label1=None,
            auto_crop_zero_pad=True,
            covisualize=True,
            # Pass original level mask for grid visualization
            crop_mask0=crop_mask0,
            crop_mask1=crop_mask1,
            covis_mask0=mask0_o,
            covis_mask1=mask1_o,
        )
    except Exception as e:
        print(f"viz_pairs: plotting failed for {name0}, {name1}: {e}")
        return

# --- New: Debug visualization helpers ---
def _local_max(mask, x, y, r=1):
    """Local max around integer (x,y) within radius r."""
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return 0.0
    if r <= 0:
        return float(mask[yi, xi])
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    return float(mask[y0:y1, x0:x1].max())


def mask_filter_and_reweight(results, thresh=0.5, dilate=1, grid_win: int = 16):
    """
    Filter matches using predicted masks at original image scale.
    """
    # If no masks at all, nothing to do
    if 'mask0' not in results and 'mask0_bin' not in results:
        return results
    if 'mask1' not in results and 'mask1_bin' not in results:
        return results

    # Use a gridified binary mask for filtering — but do NOT overwrite results
    try:
        # derive binary source for each image (prefer explicit mask*_bin, fallback to thresholded soft mask)
        if 'mask0_bin' in results and results.get('mask0_bin') is not None:
            bin0 = results['mask0_bin']
        elif 'mask0' in results and results.get('mask0') is not None:
            m0t = results['mask0']
            bin0 = (m0t > thresh).float() if isinstance(m0t, torch.Tensor) else (np.asarray(m0t) > thresh).astype(np.float32)
        else:
            return results

        if 'mask1_bin' in results and results.get('mask1_bin') is not None:
            bin1 = results['mask1_bin']
        elif 'mask1' in results and results.get('mask1') is not None:
            m1t = results['mask1']
            bin1 = (m1t > thresh).float() if isinstance(m1t, torch.Tensor) else (np.asarray(m1t) > thresh).astype(np.float32)
        else:
            return results

        # gridify (local copy)
        grid_m0 = _gridify_bin_mask(bin0, win=grid_win, thresh=thresh)
        grid_m1 = _gridify_bin_mask(bin1, win=grid_win, thresh=thresh)

        # convert to numpy masks used for local_max checks
        mask0 = _to_numpy_mask(grid_m0)
        mask1 = _to_numpy_mask(grid_m1)
    except Exception:
        # fallback to original behavior if anything fails
        if 'mask0' in results:
            mask0 = _to_numpy_mask(results['mask0'])
        else:
            mask0 = np.ones((1, 1), dtype=np.float32)
        if 'mask1' in results:
            mask1 = _to_numpy_mask(results['mask1'])
        else:
            mask1 = np.ones((1, 1), dtype=np.float32)

    # Matching.forward may store boolean in torch tensor or list
    prefiltered_flag = False
    if 'mask_prefiltered' in results:
        flag = results['mask_prefiltered']
        if isinstance(flag, (list, tuple)):
            flag = flag[0]
        if isinstance(flag, torch.Tensor):
            prefiltered_flag = bool(flag.detach().cpu().item())
        else:
            prefiltered_flag = bool(flag)

    # fetch points and indices (need idx to decide keep-array length)
    if 'index0' not in results or 'index1' not in results:
        return results
    if 'kpts0_ori' not in results or 'kpts1_ori' not in results:
        return results

    k0 = np.asarray(results['kpts0_ori'])
    k1 = np.asarray(results['kpts1_ori'])
    idx0 = np.asarray(results['index0'])
    idx1 = np.asarray(results['index1'])

    if idx0.size == 0 or idx1.size == 0:
        results['mask_postfilter_keep'] = np.array([], dtype=bool)
        return results

    mk0 = np.asarray(results.get('mkpts0', k0[idx0]))
    mk1 = np.asarray(results.get('mkpts1', k1[idx1]))
    mconf = np.asarray(results.get('mconf', results.get('conf', np.ones(len(idx0), dtype=np.float32))))

    keep = []
    for i in range(len(idx0)):
        x0, y0 = mk0[i]
        x1, y1 = mk1[i]
        m0v = _local_max(mask0, x0, y0, dilate)
        m1v = _local_max(mask1, x1, y1, dilate)
        keep.append(min(m0v, m1v) >= thresh)

    keep = np.asarray(keep, dtype=bool)
    results['mask_postfilter_keep'] = keep.copy()

    if keep.sum() == 0:
        results['index0'] = np.array([], dtype=idx0.dtype)
        results['index1'] = np.array([], dtype=idx1.dtype)
        results['mkpts0'] = np.empty((0, 2), dtype=np.float32)
        results['mkpts1'] = np.empty((0, 2), dtype=np.float32)
        results['mconf'] = np.array([], dtype=np.float32)
        results['conf'] = results['mconf']
        return results

    results['index0'] = idx0[keep]
    results['index1'] = idx1[keep]
    results['mkpts0'] = mk0[keep]
    results['mkpts1'] = mk1[keep]
    results['mconf'] = np.asarray(mconf, dtype=np.float32)[keep]
    results['conf'] = results['mconf']
    return results


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
        }, default='superglue_indoor', help='Matcher type',
    )
    parser.add_argument(
        '--extractor', choices={
            'superpoint_aachen', 'superpoint_inloc', 'd2net-ss', 'r2d2-desc',
            'context-desc', 'landmark', 'aslfeat-desc', 'disk-desc', 'swin-disk-desc',
        }, default='superpoint_aachen', help='Feature extractor type',
    )
    parser.add_argument(
        '--overlaper', choices={'oetr', 'oetr_imc', 'samatcher'},
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
    # --- New CLI flags for mask-guided filtering ---
    parser.add_argument(
        '--mask_filter', action='store_true',
        help='Filter matches by predicted masks at original scale'
    )
    parser.add_argument(
        '--mask_filter_thresh', type=float, default=0.5,
        help='Threshold on local mask probability'
    )
    parser.add_argument(
        '--mask_filter_dilate', type=int, default=2,
        help='Local window radius for mask lookup'
    )

    parser.add_argument(
        '--viz_box_mask_constraint', action='store_true',
        help='Output mask+box constraint visualization per pair'
    )
    parser.add_argument(
        '--viz_box_mask_thresh', type=float, default=0.5,
        help='Threshold used for mask rendering inside bbox'
    )

    # Removed CLI args not used by release shell workflow:
    # --mask_reweight
    # --debug_viz
    # --debug_viz_every
    # --debug_viz_max_matches

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
        mask_filter=opt.mask_filter,
        mask_filter_thresh=opt.mask_filter_thresh,
        mask_filter_dilate=opt.mask_filter_dilate,
        viz_box_mask_constraint=opt.viz_box_mask_constraint,
        viz_box_mask_thresh=opt.viz_box_mask_thresh,
    )