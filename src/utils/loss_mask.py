import torch
import torch.nn as nn
from typing import List
import src.utils.misc as misc
from torch.nn import functional as F
from src.utils.utils import box_xyxy_to_cxywh, bbox_oiou
import matplotlib.pyplot as plt
import os

class IouOverlapLoss(nn.Module):
    """IoU-based loss for bounding box regression."""
    
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, oiou=False):
        super(IouOverlapLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.oiou = oiou

    def forward(self, pred1, target1, pred2, target2, weight=None, reduction_override=None, **kwargs):
        if self.oiou:
            loss1 = oiou_loss(pred1, target1)
            loss2 = oiou_loss(pred2, target2)
        else:
            loss1 = giou_loss(pred1, target1)
            loss2 = giou_loss(pred2, target2)
        return (loss1 + loss2) / 2.0


def extract_mask_bbox(masks, threshold=0.5, use_sigmoid=True):
    """Extract bounding boxes from mask predictions."""
    batch_size = masks.shape[0]
    device = masks.device
    bboxes = []
    
    for i in range(batch_size):
        if use_sigmoid:
            mask = (masks[i, 0].sigmoid() > threshold).float()
        else:
            mask = (masks[i, 0] > threshold).float()
        
        if mask.sum() == 0:
            # Use full image as bbox if no pixels above threshold
            h, w = mask.shape
            bbox = torch.tensor([0, 0, w-1, h-1], device=device, dtype=torch.float32)
        else:
            # Find bounding box from non-zero pixels
            coords = torch.nonzero(mask, as_tuple=False).float()
            y_coords, x_coords = coords[:, 0], coords[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            bbox = torch.stack([x_min, y_min, x_max, y_max])
        
        bboxes.append(bbox)
    
    return torch.stack(bboxes)


def create_box_mask(boxes, mask_shape, device):
    """Create binary masks from bounding boxes for region-aware loss."""
    h, w = mask_shape
    batch_size = boxes.shape[0]
    box_masks = torch.zeros((batch_size, h, w), device=device)
    
    for i in range(batch_size):
        x1, y1, x2, y2 = boxes[i].clamp(min=0)
        x1, y1 = int(x1.item()), int(y1.item())
        x2, y2 = min(int(x2.item()), w-1), min(int(y2.item()), h-1)
        
        if x2 > x1 and y2 > y1:
            box_masks[i, y1:y2+1, x1:x2+1] = 1.0
    
    return box_masks


def region_aware_mask_loss(pred_masks, gt_masks, pred_boxes, num_masks, 
                          in_box_weight=2.0, out_box_weight=0.5, visualize=False, save_dir=None):
    """Compute region-aware mask loss with proper normalization."""
    h, w = pred_masks.shape[2], pred_masks.shape[3]
    
    # Create weighted masks: higher weight inside box, lower outside
    box_masks = create_box_mask(pred_boxes, (h, w), pred_masks.device)
    weight_masks = box_masks * in_box_weight + (1 - box_masks) * out_box_weight
    
    # Visualization code
    if visualize:
        visualize_weight_masks(weight_masks, pred_masks, gt_masks, pred_boxes, save_dir)
    
    # Flatten for loss computation
    pred_flat = pred_masks.sigmoid().flatten(1)
    gt_flat = gt_masks.flatten(1)
    weight_flat = weight_masks.flatten(1)
    
    # Weighted BCE loss with normalization by weight sum
    bce_loss = F.binary_cross_entropy(pred_flat, gt_flat, reduction='none')
    weighted_bce = (bce_loss * weight_flat).sum(-1)
    weight_sum = weight_flat.sum(-1).clamp(min=1e-6)
    normalized_bce = weighted_bce / weight_sum
    
    # Weighted Dice loss
    intersection = (pred_flat * gt_flat * weight_flat).sum(-1)
    pred_weighted = (pred_flat * weight_flat).sum(-1)
    gt_weighted = (gt_flat * weight_flat).sum(-1)
    dice_loss = 1 - (2 * intersection + 1) / (pred_weighted + gt_weighted + 1)
    
    return normalized_bce.sum() / num_masks, dice_loss.sum() / num_masks


def visualize_weight_masks(weight_masks, pred_masks, gt_masks, pred_boxes, save_dir=None):
    """Visualize weight masks for each sample in the batch."""
    if save_dir is None:
        save_dir = "./weight_mask_visualizations"
    
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = weight_masks.shape[0]
    
    for i in range(batch_size):
        # Convert tensors to numpy for visualization
        weight_mask = weight_masks[i].detach().cpu().numpy()
        pred_mask = pred_masks[i, 0].sigmoid().detach().cpu().numpy()
        gt_mask = gt_masks[i, 0].detach().cpu().numpy()
        
        # Get box coordinates
        x1, y1, x2, y2 = pred_boxes[i].detach().cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sample {i} - Weight Mask Visualization', fontsize=16)
        
        # 1. Weight mask
        im1 = axes[0, 0].imshow(weight_mask, cmap='viridis', interpolation='nearest')
        axes[0, 0].set_title('Weight Mask')
        axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='red', linewidth=2))
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. Predicted mask
        im2 = axes[0, 1].imshow(pred_mask, cmap='gray', interpolation='nearest')
        axes[0, 1].set_title('Predicted Mask')
        axes[0, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='red', linewidth=2))
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Ground truth mask
        im3 = axes[1, 0].imshow(gt_mask, cmap='gray', interpolation='nearest')
        axes[1, 0].set_title('Ground Truth Mask')
        axes[1, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='red', linewidth=2))
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 4. Weight distribution histogram
        axes[1, 1].hist(weight_mask.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Weight Distribution')
        axes[1, 1].set_xlabel('Weight Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(weight_mask.mean(), color='red', linestyle='--', 
                          label=f'Mean: {weight_mask.mean():.3f}')
        axes[1, 1].legend()
        
        # Add text info
        info_text = f'Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})\n'
        info_text += f'Weight range: [{weight_mask.min():.2f}, {weight_mask.max():.2f}]\n'
        info_text += f'In-box pixels: {(weight_mask == weight_mask.max()).sum()}\n'
        info_text += f'Out-box pixels: {(weight_mask == weight_mask.min()).sum()}'
        
        fig.text(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'weight_mask_sample_{i}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Weight mask visualizations saved to: {save_dir}")


def visualize_weight_masks_single_plot(weight_masks, pred_masks, gt_masks, pred_boxes, save_dir=None):
    """Visualize all weight masks in a single plot for comparison."""
    if save_dir is None:
        save_dir = "./weight_mask_visualizations"
    
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = weight_masks.shape[0]
    cols = min(4, batch_size)
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if batch_size == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(batch_size):
        weight_mask = weight_masks[i].detach().cpu().numpy()
        x1, y1, x2, y2 = pred_boxes[i].detach().cpu().numpy()
        
        im = axes[i].imshow(weight_mask, cmap='viridis', interpolation='nearest')
        axes[i].set_title(f'Sample {i}')
        axes[i].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, edgecolor='red', linewidth=2))
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Add weight range text
        axes[i].text(0.02, 0.98, f'Range: [{weight_mask.min():.2f}, {weight_mask.max():.2f}]',
                    transform=axes[i].transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')
    
    # Hide empty subplots
    for i in range(batch_size, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_masks_batch_overview.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Batch weight mask overview saved to: {save_dir}")


def oiou_loss(pred, target, eps=1e-7):
    """Orientation-aware IoU loss."""
    ious = bbox_oiou(target, pred, eps)
    loss = 1 - ious
    return loss


def giou_loss(pred, target, eps=1e-7):
    """Generalized IoU loss for bounding box regression."""
    # Intersection
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # Union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # Enclosing area for GIoU
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

    # GIoU calculation
    gious = ious - (enclose_area - union) / enclose_area
    loss = 1 - gious
    return loss


def point_sample(input, point_coords, **kwargs):
    """Sample features at given coordinates using bilinear interpolation."""
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """Efficient tensor concatenation that avoids copy for single element."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """Sample points based on prediction uncertainty for PointRend-style training."""
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    
    # Calculate uncertainty and select most uncertain points
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    
    # Add random points if needed
    if num_random_points > 0:
        point_coords = cat([
            point_coords,
            torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
        ], dim=1)
    return point_coords


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """Compute DICE loss for mask predictions."""
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """Compute sigmoid cross-entropy loss for mask predictions."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


def calculate_uncertainty(logits):
    """Calculate uncertainty as L1 distance from zero for foreground class."""
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):
    """Compute point-based mask losses using uncertainty sampling."""
    with torch.no_grad():
        # Sample uncertain points for training
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # Sample ground truth labels at selected points
        point_labels = point_sample(
            target_masks, point_coords, align_corners=False
        ).squeeze(1)

    # Sample predictions at selected points
    point_logits = point_sample(
        src_masks, point_coords, align_corners=False
    ).squeeze(1)

    # Compute losses
    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

    del src_masks
    del target_masks
    return loss_mask, loss_dice


def loss(batch, mask_box_weight_ratio=1.0, box_loss_weights=None, mask_loss_config=None, visualize_weights=False):
    """
    Compute Mask-Centric joint optimization loss with proper coordinate handling.
    
    Args:
        batch: Input batch data
        mask_box_weight_ratio: Weight ratio between mask loss and box loss
        box_loss_weights: Dict with weights for different box loss components
        mask_loss_config: Dict with configuration for mask loss
        visualize_weights: Whether to visualize weight masks
    """
    if box_loss_weights is None:
        box_loss_weights = {
            'iou_weight': 2.0,
            'loc_weight': 1.0,
            'wh_weight': 1.5,
            'consistency_weight': 1.0
        }
    
    if mask_loss_config is None:
        mask_loss_config = {
            'point_weight': 1.0,
            'region_weight': 1.5,
            'in_box_weight': 2.0,
            'out_box_weight': 0.5
        }
    
    loss_scalars = {}
    gt_masks = batch["gt_masks"]
    pred_mask = batch["pred_mask"]
    h_p, w_p = pred_mask.shape[2], pred_mask.shape[3]
    
    # Resize ground truth masks to prediction resolution (pure bilinear, no morphology)
    gt_masks_r = F.interpolate(gt_masks, size=(h_p, w_p), mode='bilinear', align_corners=False)
    
    # Get original mask dimensions
    h0, w0 = gt_masks[:, 0].shape[1], gt_masks[:, 0].shape[2]
    h1, w1 = gt_masks[:, 1].shape[1], gt_masks[:, 1].shape[2]

    # Ground truth and predicted bounding boxes
    gt_bbox_xyxy0 = batch['bbox0']
    gt_bbox_xyxy1 = batch['bbox1']
    gt_bbox_cxywh0 = box_xyxy_to_cxywh(gt_bbox_xyxy0, max_h=h0, max_w=w0)
    gt_bbox_cxywh1 = box_xyxy_to_cxywh(gt_bbox_xyxy1, max_h=h1, max_w=w1)

    pred_bbox_xyxy0, pred_bbox_xyxy1 = batch["boxes"]
    pred_bbox_cxywh0 = box_xyxy_to_cxywh(pred_bbox_xyxy0, max_h=h0, max_w=w0)
    pred_bbox_cxywh1 = box_xyxy_to_cxywh(pred_bbox_xyxy1, max_h=h1, max_w=w1)
    
    # ===== MASK LOSS COMPUTATION =====
    
    # 1. Original point-based mask losses
    point_loss_mask0, point_loss_dice0 = loss_masks(
        pred_mask[:, 0, :, :].unsqueeze(1), 
        gt_masks_r[:, 0, :, :].unsqueeze(1), 
        len(pred_mask)
    )
    point_loss_mask1, point_loss_dice1 = loss_masks(
        pred_mask[:, 1, :, :].unsqueeze(1), 
        gt_masks_r[:, 1, :, :].unsqueeze(1), 
        len(pred_mask)
    )
    point_loss_mask = (point_loss_mask0 + point_loss_mask1) / 2
    point_loss_dice = (point_loss_dice0 + point_loss_dice1) / 2
    original_mask_loss = point_loss_mask + point_loss_dice
    
    # 2. Region-aware mask losses with visualization
    # Scale predicted boxes to mask resolution
    scale_x0, scale_y0 = w_p / w0, h_p / h0
    scale_x1, scale_y1 = w_p / w1, h_p / h1
    
    scaled_pred_bbox0 = pred_bbox_xyxy0 * torch.tensor([scale_x0, scale_y0, scale_x0, scale_y0], device=pred_bbox_xyxy0.device)
    scaled_pred_bbox1 = pred_bbox_xyxy1 * torch.tensor([scale_x1, scale_y1, scale_x1, scale_y1], device=pred_bbox_xyxy1.device)
    
    region_loss_mask0, region_loss_dice0 = region_aware_mask_loss(
        pred_mask[:, 0, :, :].unsqueeze(1),
        gt_masks_r[:, 0, :, :].unsqueeze(1),
        scaled_pred_bbox0,
        len(pred_mask),
        mask_loss_config['in_box_weight'],
        mask_loss_config['out_box_weight'],
        visualize=visualize_weights,
        save_dir=".data/visualizations/mask0_weights"
    )
    
    region_loss_mask1, region_loss_dice1 = region_aware_mask_loss(
        pred_mask[:, 1, :, :].unsqueeze(1),
        gt_masks_r[:, 1, :, :].unsqueeze(1),
        scaled_pred_bbox1,
        len(pred_mask),
        mask_loss_config['in_box_weight'],
        mask_loss_config['out_box_weight'],
        visualize=visualize_weights,
        save_dir=".data/visualizations/mask1_weights"
    )
    
    region_mask_loss = (region_loss_mask0 + region_loss_mask1) / 2
    region_dice_loss = (region_loss_dice0 + region_loss_dice1) / 2
    region_aware_loss = region_mask_loss + region_dice_loss
    
    # Combined mask loss
    total_mask_loss = (mask_loss_config['point_weight'] * original_mask_loss + 
                      mask_loss_config['region_weight'] * region_aware_loss)
    
    loss_scalars.update({
        "loss_mask_point": original_mask_loss.item(),
        "loss_mask_region": region_aware_loss.item(),
        "loss_mask_total": total_mask_loss.item()
    })
    
    # ===== BOX LOSS COMPUTATION =====
    
    # Scale factors for normalization
    wh_scale0 = torch.tensor([w0, h0], device=gt_masks.device)
    wh_scale1 = torch.tensor([w1, h1], device=gt_masks.device)

    # 1. Localization L1 loss for box centers
    loc_l1_loss = (F.l1_loss(
        pred_bbox_cxywh0[:, :2] / wh_scale0, gt_bbox_cxywh0[:, :2] / wh_scale0, reduction='mean'
    ) + F.l1_loss(
        pred_bbox_cxywh1[:, :2] / wh_scale1, gt_bbox_cxywh1[:, :2] / wh_scale1, reduction='mean'
    )) / 2

    # 2. L1 loss for box dimensions
    wh_l1_loss = (F.l1_loss(
        pred_bbox_cxywh0[:, 2:] / wh_scale0, gt_bbox_cxywh0[:, 2:] / wh_scale0, reduction='mean'
    ) + F.l1_loss(
        pred_bbox_cxywh1[:, 2:] / wh_scale1, gt_bbox_cxywh1[:, 2:] / wh_scale1, reduction='mean'
    )) / 2

    # 3. IoU-based loss
    iouloss_func = IouOverlapLoss(reduction='mean', oiou=False)
    iouloss_per_sample = iouloss_func(pred_bbox_xyxy0, gt_bbox_xyxy0, pred_bbox_xyxy1, gt_bbox_xyxy1)
    mean_iouloss = iouloss_per_sample.mean()
    
    # 4. Mask-Box consistency loss (geometric consistency constraint)
    # Extract bboxes from predicted masks
    mask_bbox0 = extract_mask_bbox(pred_mask[:, 0, :, :].unsqueeze(1), use_sigmoid=True)
    mask_bbox1 = extract_mask_bbox(pred_mask[:, 1, :, :].unsqueeze(1), use_sigmoid=True)
    
    # Convert mask-derived boxes to original image coordinates
    mask_to_orig_scale_x0 = w0 / w_p
    mask_to_orig_scale_y0 = h0 / h_p
    mask_to_orig_scale_x1 = w1 / w_p
    mask_to_orig_scale_y1 = h1 / h_p
    
    mask_bbox0_orig_coords = mask_bbox0 * torch.tensor(
        [mask_to_orig_scale_x0, mask_to_orig_scale_y0, mask_to_orig_scale_x0, mask_to_orig_scale_y0], 
        device=mask_bbox0.device
    )
    mask_bbox1_orig_coords = mask_bbox1 * torch.tensor(
        [mask_to_orig_scale_x1, mask_to_orig_scale_y1, mask_to_orig_scale_x1, mask_to_orig_scale_y1], 
        device=mask_bbox1.device
    )
    
    # Normalize consistency loss by image dimensions to avoid scale sensitivity
    norm_pred_bbox0 = pred_bbox_xyxy0 / torch.tensor([w0, h0, w0, h0], device=pred_bbox_xyxy0.device)
    norm_mask_bbox0 = mask_bbox0_orig_coords / torch.tensor([w0, h0, w0, h0], device=mask_bbox0_orig_coords.device)
    norm_pred_bbox1 = pred_bbox_xyxy1 / torch.tensor([w1, h1, w1, h1], device=pred_bbox_xyxy1.device)
    norm_mask_bbox1 = mask_bbox1_orig_coords / torch.tensor([w1, h1, w1, h1], device=mask_bbox1_orig_coords.device)
    
    consistency_loss = (F.l1_loss(norm_pred_bbox0, norm_mask_bbox0, reduction='mean') + 
                       F.l1_loss(norm_pred_bbox1, norm_mask_bbox1, reduction='mean')) / 2
    
    # Weighted box loss combination
    box_loss = (box_loss_weights['iou_weight'] * mean_iouloss + 
                box_loss_weights['wh_weight'] * wh_l1_loss + 
                box_loss_weights['loc_weight'] * loc_l1_loss + 
                box_loss_weights['consistency_weight'] * consistency_loss)

    loss_scalars.update({
        "loss_iou": mean_iouloss.item(),
        "loss_wh": wh_l1_loss.item(), 
        "loss_loc": loc_l1_loss.item(),
        "loss_consistency": consistency_loss.item(),
        "loss_box_total": box_loss.item()
    })

    # ===== TOTAL LOSS =====
    # Apply mask-box weight ratio
    mask_weight = mask_box_weight_ratio / (1 + mask_box_weight_ratio)
    box_weight = 1 / (1 + mask_box_weight_ratio)
    
    total_loss = mask_weight * total_mask_loss + box_weight * box_loss
    
    loss_scalars.update({
        "total_loss": total_loss.item()
    })
    
    batch.update({"loss": total_loss, "loss_scalars": loss_scalars})
