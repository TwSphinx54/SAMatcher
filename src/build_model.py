import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from loguru import logger
from pathlib import Path # Ensure pathlib is imported
from src.utils.loss_mask import loss
from src.utils.utils import draw_bbox
from src.utils.metrics_mask import metrics
from src.modeling.sam2_base import SAM2Base
from src.utils.transforms import SAM2Transforms
from src.utils.profiler import PassThroughProfiler


class ModelTrainer():
    def __init__(self, config, sam_model: SAM2Base, pretrained_ckpt=None, profiler=None, dump_dir=None, accelerator=None,
                 mask_threshold=0.0,
                 max_hole_area=0.0,
                 max_sprinkle_area=0.0,
                 **kwargs) -> None:
        """Initializes the ModelTrainer.

        Args:
            config: Configuration object for the trainer and model.
            sam_model (SAM2Base): The SAM-2 model instance.
            pretrained_ckpt (str, optional): Path to a pretrained checkpoint. Defaults to None.
            profiler (Profiler, optional): Profiler instance. Defaults to PassThroughProfiler.
            dump_dir (str, optional): Directory for dumping outputs (e.g., visualizations). Defaults to None.
            accelerator (Accelerator, optional): Accelerator instance for distributed training. Defaults to None.
            mask_threshold (float, optional): Threshold for converting mask logits to binary masks.
                Used by SAM2Transforms. Defaults to 0.0.
            max_hole_area (float, optional): Maximum area for filling holes in masks.
                Used by SAM2Transforms. Defaults to 0.0.
            max_sprinkle_area (float, optional): Maximum area for removing sprinkles in masks.
                Used by SAM2Transforms. Defaults to 0.0.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        if isinstance(sam_model, torch.nn.parallel.DistributedDataParallel):
            self.model = sam_model.module
        else:
            self.model = sam_model

        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None  # List of (height, width) tuples
        self._is_batch = False  # Whether the predictor is set for a batch of images

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps (example, adjust if necessary)
        self._bb_feat_sizes = [
            (self.model.image_size // 4, self.model.image_size // 4),  # Example, adjust based on actual model
            (self.model.image_size // 8, self.model.image_size // 8),
            (self.model.image_size // 16, self.model.image_size // 16),
        ]

        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.accelerator = accelerator
        if self.accelerator:
            self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // self.accelerator.num_processes, 1)
        else:
            self.n_vals_plot = config.TRAINER.N_VAL_PAIRS_TO_PLOT

        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')
            if 'state_dict' in state_dict:
                actual_state_dict = state_dict['state_dict']
                # Remove "model." prefix if present (common in some training frameworks)
                actual_state_dict = {k.replace("model.", ""): v for k, v in actual_state_dict.items()}
            else:
                actual_state_dict = state_dict

            self.model.load_state_dict(actual_state_dict, strict=True)
            logger.info(f"Loaded '{pretrained_ckpt}' as pretrained checkpoint")

        self.dump_dir = dump_dir

    def _trainval_inference(self, batch: dict):
        """
        Performs inference during training or validation.
        Updates the batch dictionary with predictions.

        Args:
            batch (dict): A dictionary containing input data (e.g., 'image0', 'image1').
        """
        with self.profiler.profile("Segmenter"):
            image0 = batch['image0'].unsqueeze(0) if batch['image0'].ndim == 3 else batch['image0']
            image1 = batch['image1'].unsqueeze(0) if batch['image1'].ndim == 3 else batch['image1']

            mask, boxes = self.model(image0, image1)
            batch.update({
                'pred_mask': mask,
                'boxes': boxes
            })

        with self.profiler.profile("Compute losses"):
            loss(batch)

    def _compute_metrics(self, batch: dict) -> dict:
        """
        Computes evaluation metrics for the given batch.

        Args:
            batch (dict): A dictionary containing ground truth and predictions.

        Returns:
            dict: A dictionary containing computed metrics.
        """
        with self.profiler.profile("Compute metrics"):
            all_metrics = metrics(batch)
            batch.update(all_metrics)

            metrics_to_log = {}

            for key, value_tensor in all_metrics.items():
                metrics_to_log[key] = value_tensor

            ret_dict = {'metrics': metrics_to_log}
        return ret_dict

    def _visualize_batch(self, batch: dict, outputs: dict, batch_idx: int, prefix: str = "val", max_samples: int = 4, global_step: int = 0):
        """
        Visualizes a batch of data and predictions and saves them locally.

        Args:
            batch (dict): The input batch data.
            outputs (dict): The model outputs for the batch.
            batch_idx (int): The index of the current batch.
            prefix (str, optional): Prefix for filenames and subdirectories (e.g., "val_ds0_epoch0"). Defaults to "val".
            max_samples (int, optional): Maximum number of samples to visualize from the batch. Defaults to 4.
            global_step (int, optional): The global training step, used in filenames. Defaults to 0.
        """
        if not self.accelerator or not self.accelerator.is_main_process:
            return

        if not self.dump_dir:
            logger.info(f"self.dump_dir is not set in ModelTrainer. Skipping local saving of visualizations for {prefix}, batch {batch_idx}.")
            return

        gt_mask_color = (0, 255, 0)
        pred_mask_color = (0, 0, 255)
        text_color = (255, 255, 255)
        font_scale = 0.5
        thickness = 1
        alpha = 0.4

        images0 = batch['image0']
        images1 = batch['image1']
        gt_masks_batch = batch['gt_masks']
        gt_bboxes0_batch = batch.get('bbox0')
        gt_bboxes1_batch = batch.get('bbox1')

        model_boxes_output = outputs.get('boxes')
        pred_bboxes0_model_output_batch = None
        pred_bboxes1_model_output_batch = None
        if isinstance(model_boxes_output, (tuple, list)) and len(model_boxes_output) >= 2:
            pred_bboxes0_model_output_batch = model_boxes_output[0]
            pred_bboxes1_model_output_batch = model_boxes_output[1]

        num_samples_to_vis = min(max_samples, images0.size(0), self.n_vals_plot)
        B, C, H, W = images0.shape

        pred_mask_logits_batch = outputs['pred_mask']

        pred_masks_processed = F.interpolate(
            pred_mask_logits_batch, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        pred_masks_binary_batch = (torch.sigmoid(pred_masks_processed) > 0.5).float()

        for i in range(num_samples_to_vis):
            img0_tensor = images0[i]
            img1_tensor = images1[i]
            # Convert RGB to BGR for OpenCV
            img0_np_orig = (img0_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img0_np_orig = cv.cvtColor(img0_np_orig, cv.COLOR_RGB2BGR)
            img1_np_orig = (img1_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img1_np_orig = cv.cvtColor(img1_np_orig, cv.COLOR_RGB2BGR)

            vis_bbox_pair = []
            vis_mask_pair = []

            items_data_for_sample = [
                {
                    "name": "img0",
                    "img_np_orig": img0_np_orig,
                    "gt_mask_ch": gt_masks_batch[i, 0],
                    "pred_mask_ch": pred_masks_binary_batch[i, 0],
                    "gt_bbox_tensor": gt_bboxes0_batch[i] if gt_bboxes0_batch is not None else None,
                    "pred_bbox_tensor": pred_bboxes0_model_output_batch[i] if pred_bboxes0_model_output_batch is not None else None,
                },
                {
                    "name": "img1",
                    "img_np_orig": img1_np_orig,
                    "gt_mask_ch": gt_masks_batch[i, 1],
                    "pred_mask_ch": pred_masks_binary_batch[i, 1],
                    "gt_bbox_tensor": gt_bboxes1_batch[i] if gt_bboxes1_batch is not None else None,
                    "pred_bbox_tensor": pred_bboxes1_model_output_batch[i] if pred_bboxes1_model_output_batch is not None else None,
                }
            ]

            for item_data in items_data_for_sample:
                img_for_bbox = item_data["img_np_orig"].copy()
                gt_bbox_coords_tensor = item_data["gt_bbox_tensor"]
                pred_bbox_coords_sample_tensor = item_data["pred_bbox_tensor"]

                gt_bbox_np = None
                if gt_bbox_coords_tensor is not None and gt_bbox_coords_tensor.nelement() > 0:
                    gt_bbox_np = gt_bbox_coords_tensor.cpu().numpy().astype(np.int32).reshape(4)

                pred_bbox_np = None
                if pred_bbox_coords_sample_tensor is not None and pred_bbox_coords_sample_tensor.nelement() > 0:
                    pred_bbox_np = pred_bbox_coords_sample_tensor.detach().cpu().numpy().astype(np.int32).reshape(4)

                img_for_bbox = draw_bbox(img_for_bbox, box=pred_bbox_np, gt_box=gt_bbox_np, score=None)
                vis_bbox_pair.append(img_for_bbox)

                img_for_mask = item_data["img_np_orig"].copy()
                gt_mask_ch = item_data["gt_mask_ch"]
                pred_mask_ch = item_data["pred_mask_ch"]

                gt_m_np = gt_mask_ch.cpu().numpy().astype(np.uint8)
                colored_gt_mask_overlay = np.zeros_like(img_for_mask, dtype=np.uint8)
                colored_gt_mask_overlay[gt_m_np == 1] = gt_mask_color
                img_for_mask = cv.addWeighted(img_for_mask, 1, colored_gt_mask_overlay, alpha, 0)

                pred_m_np = pred_mask_ch.cpu().numpy().astype(np.uint8)
                colored_pred_mask_overlay = np.zeros_like(img_for_mask, dtype=np.uint8)
                colored_pred_mask_overlay[pred_m_np == 1] = pred_mask_color
                img_for_mask = cv.addWeighted(img_for_mask, 1, colored_pred_mask_overlay, alpha, 0)

                iou = self._calculate_single_iou(gt_mask_ch, pred_mask_ch)
                cv.putText(img_for_mask, f"IoU: {iou:.3f}", (5, 15), cv.FONT_HERSHEY_SIMPLEX,
                           font_scale, text_color, thickness)

                vis_mask_pair.append(img_for_mask)

            # Create subdirectory for the current prefix (e.g., val_ds0_epoch0) if it doesn't exist
            save_path_prefix_dir = Path(self.dump_dir) / prefix
            try:
                save_path_prefix_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {save_path_prefix_dir}: {e}")
                continue # Skip saving for this sample if directory creation fails

            if len(vis_bbox_pair) == 2:
                final_vis_bbox_img = np.concatenate(vis_bbox_pair, axis=1)
                try:
                    bbox_filename = f"sample_{batch_idx}_{i}_step_{global_step}_bboxes.png"
                    bbox_save_path = save_path_prefix_dir / bbox_filename
                    cv.imwrite(str(bbox_save_path), final_vis_bbox_img)
                    # logger.debug(f"Saved bbox visualization to {bbox_save_path}") 
                except Exception as e:
                    logger.error(f"Failed to save bbox visualization for {prefix}, sample {batch_idx}_{i} to {bbox_save_path}: {e}")

            if len(vis_mask_pair) == 2:
                final_vis_mask_img = np.concatenate(vis_mask_pair, axis=1)
                try:
                    mask_filename = f"sample_{batch_idx}_{i}_step_{global_step}_masks.png"
                    mask_save_path = save_path_prefix_dir / mask_filename
                    cv.imwrite(str(mask_save_path), final_vis_mask_img)
                    # logger.debug(f"Saved mask visualization to {mask_save_path}")
                except Exception as e:
                    logger.error(f"Failed to save mask visualization for {prefix}, sample {batch_idx}_{i} to {mask_save_path}: {e}")

    def _calculate_single_iou(self, gt_mask_tensor: torch.Tensor, pred_mask_tensor: torch.Tensor) -> float:
        """
        Calculates IoU for a single pair of ground truth and prediction masks.

        Args:
            gt_mask_tensor (torch.Tensor): Ground truth mask (binary or boolean).
            pred_mask_tensor (torch.Tensor): Predicted mask (binary or boolean).

        Returns:
            float: The IoU score.
        """
        intersection = torch.logical_and(gt_mask_tensor.bool(), pred_mask_tensor.bool()).sum().float()
        union = torch.logical_or(gt_mask_tensor.bool(), pred_mask_tensor.bool()).sum().float()
        iou = intersection / union if union > 0 else 0.0
        return iou.item()