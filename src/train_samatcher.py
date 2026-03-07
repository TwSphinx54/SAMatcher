# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from src.modeling.sam2_base import SAM2Base
from src.build_samatcher import SAM2ImagePredictor
from src.utils.transforms import SAM2Transforms

class SAM2Dataset(Dataset):
    """Dataset for training SAM2 model."""
    
    def __init__(self, 
                 image_paths: List[str], 
                 mask_paths: List[str],
                 transform=None):
        """
        Args:
            image_paths: List of paths to input images
            mask_paths: List of paths to corresponding ground truth masks
            transform: Optional transforms to apply
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = np.array(np.load(self.image_paths[idx]))
        
        # Load mask
        mask = np.array(np.load(self.mask_paths[idx]))
        
        # Generate random prompts (points, boxes) for training
        prompts = self._generate_prompts(mask)
        
        sample = {
            'image': image,
            'mask': mask,
            'prompts': prompts
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _generate_prompts(self, mask):
        """Generate random prompts (points/boxes) from ground truth mask."""
        # Get mask properties
        pos_points = []
        neg_points = []
        box = None
        
        # Generate positive/negative points and bounding box
        if np.max(mask) > 0:
            # For positive points (foreground)
            pos_indices = np.where(mask > 0)
            if len(pos_indices[0]) > 0:
                rand_idx = np.random.choice(len(pos_indices[0]), size=min(3, len(pos_indices[0])), replace=False)
                pos_points = [[pos_indices[1][i], pos_indices[0][i]] for i in rand_idx]
            
            # For negative points (background)
            neg_indices = np.where(mask == 0)
            if len(neg_indices[0]) > 0:
                rand_idx = np.random.choice(len(neg_indices[0]), size=min(3, len(neg_indices[0])), replace=False)
                neg_points = [[neg_indices[1][i], neg_indices[0][i]] for i in rand_idx]
            
            # Generate bounding box
            if len(pos_indices[0]) > 0:
                y_min, y_max = np.min(pos_indices[0]), np.max(pos_indices[0])
                x_min, x_max = np.min(pos_indices[1]), np.max(pos_indices[1])
                box = [x_min, y_min, x_max, y_max]
        
        return {
            'point_coords': np.array(pos_points + neg_points) if pos_points or neg_points else None,
            'point_labels': np.array([1] * len(pos_points) + [0] * len(neg_points)) if pos_points or neg_points else None,
            'box': np.array(box) if box is not None else None
        }

class SAM2Loss(nn.Module):
    """Loss function for SAM2 model."""
    
    def __init__(self, dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
    def forward(self, pred_masks, gt_masks, iou_predictions=None, gt_ious=None):
        """
        Calculate loss between predicted and ground truth masks.
        
        Args:
            pred_masks: Predicted masks (B, C, H, W)
            gt_masks: Ground truth masks (B, H, W)
            iou_predictions: Predicted IoU scores (optional)
            gt_ious: Ground truth IoU scores (optional)
        """
        # Convert ground truth to one-hot encoding
        gt_masks = gt_masks.unsqueeze(1)
        
        # Calculate dice loss
        intersection = (pred_masks * gt_masks).sum(dim=(2, 3))
        union = pred_masks.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        
        # Calculate BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_masks, gt_masks, reduction="none"
        ).mean(dim=(2, 3))
        
        # Combine losses
        loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        # Add IoU prediction loss if available
        if iou_predictions is not None and gt_ious is not None:
            iou_loss = nn.functional.mse_loss(iou_predictions, gt_ious)
            loss = loss + 0.3 * iou_loss
            
        return loss.mean()

class SAM2Trainer:
    """Trainer for SAM2 model."""
    
    def __init__(
        self,
        model: SAM2Base,
        device: torch.device = None,
        optimizer_params: Dict = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The SAM2 model to train
            device: Device to use for training
            optimizer_params: Parameters for the optimizer
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-5, "weight_decay": 1e-4}
        self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_params)
        
        # Initialize loss function
        self.criterion = SAM2Loss()
        
        # Setup predictor for inference
        self.predictor = SAM2ImagePredictor(model)
        
        # Setup checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_step(self, batch):
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        
        batch_size = images.size(0)
        losses = []
        
        for i in range(batch_size):
            # Set the current image
            self.predictor.set_image(images[i].cpu().numpy())
            
            # Get prompts
            point_coords = batch["prompts"]["point_coords"][i]
            point_labels = batch["prompts"]["point_labels"][i]
            box = batch["prompts"]["box"][i]
            
            # Predict masks
            pred_masks, iou_predictions, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
                return_logits=True
            )
            
            # Calculate loss
            loss = self.criterion(
                torch.tensor(pred_masks).to(self.device), 
                masks[i],
                torch.tensor(iou_predictions).to(self.device),
                None  # No ground truth IoU available
            )
            
            losses.append(loss)
            
        # Average loss and backpropagate
        avg_loss = torch.stack(losses).mean()
        avg_loss.backward()
        self.optimizer.step()
        
        return avg_loss.item()
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                batch_size = images.size(0)
                
                for i in range(batch_size):
                    # Set the current image
                    self.predictor.set_image(images[i].cpu().numpy())
                    
                    # Get prompts
                    point_coords = batch["prompts"]["point_coords"][i]
                    point_labels = batch["prompts"]["point_labels"][i]
                    box = batch["prompts"]["box"][i]
                    
                    # Predict masks
                    pred_masks, iou_predictions, _ = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        multimask_output=False,
                        return_logits=True
                    )
                    
                    # Calculate loss
                    loss = self.criterion(
                        torch.tensor(pred_masks).to(self.device), 
                        masks[i],
                        torch.tensor(iou_predictions).to(self.device),
                        None  # No ground truth IoU available
                    )
                    
                    val_losses.append(loss.item())
        
        return sum(val_losses) / len(val_losses)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_interval: int = 1,
    ):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            save_interval: Interval to save checkpoints
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            train_losses = []
            for batch_idx, batch in enumerate(train_loader):
                loss = self.train_step(batch)
                train_losses.append(loss)
                
                if batch_idx % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss:.4f}")
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model.pt")
                    self.logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, filename):
        """Save a checkpoint of the model."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Update predictor
        self.predictor = SAM2ImagePredictor(self.model)

# Example usage
if __name__ == "__main__":
    from src.build_sam import build_sam2
    
    # Setup model
    checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
    model = build_sam2(model_cfg, checkpoint)
    
    # Initialize trainer
    trainer = SAM2Trainer(model)
    
    # Create dummy dataset
    # In a real scenario, you would use your actual dataset paths
    image_paths = ["path/to/image1.npy", "path/to/image2.npy"]
    mask_paths = ["path/to/mask1.npy", "path/to/mask2.npy"]
    
    dataset = SAM2Dataset(image_paths, mask_paths)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Train the model
    trainer.train(train_loader, num_epochs=5)
