import torch
import src.utils.misc as misc
import torch.nn.functional as F

def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def metrics(batch):
    gt_masks = batch["gt_masks"]    # Ground truth masks, e.g., [B, 2, H, W] for pairs
    gt_masks_r = torch.cat((gt_masks[:, 0, :, :], gt_masks[:, 1, :, :]), dim=1).unsqueeze(1)
    pred_mask = batch["pred_mask"]  # Predicted masks
    pred_mask_r = F.interpolate(pred_mask, size=gt_masks_r.size()[2:], mode='bilinear', align_corners=False)
    batch["pred_mask"] = pred_mask_r

    # ------ MASK METRICS ------ #
    mask_iou = compute_iou(pred_mask_r, gt_masks_r) * 100
    boundary_iou = compute_boundary_iou(pred_mask_r, gt_masks_r) * 100

    # --- Bounding Box IoU/OIoU Metrics ---
    pred_bbox_xyxy0 = batch['boxes'][0].float()
    pred_bbox_xyxy1 = batch['boxes'][1].float()
    gt_bbox_xyxy0 = batch['bbox0'].float()
    gt_bbox_xyxy1 = batch['bbox1'].float()

    box_iou0 = misc.bbox_overlaps(pred_bbox_xyxy0, gt_bbox_xyxy0, is_aligned=True).mean() * 100
    box_iou1 = misc.bbox_overlaps(pred_bbox_xyxy1, gt_bbox_xyxy1, is_aligned=True).mean() * 100
    box_iou = (box_iou0 + box_iou1) / 2
    box_oiou0 = misc.bbox_oiou(gt_bbox_xyxy0, pred_bbox_xyxy0).mean() * 100
    box_oiou1 = misc.bbox_oiou(gt_bbox_xyxy1, pred_bbox_xyxy1).mean() * 100
    box_oiou = (box_oiou0 + box_oiou1) / 2

    metrics_results = dict(
        val_m_iou=mask_iou, 
        val_mb_iou=boundary_iou, 
        val_b_iou=box_iou,
        val_b_oiou=box_oiou,
        )

    return metrics_results