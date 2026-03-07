#!/usr/bin/env python
"""
Landmark feature extractor using SIFT keypoints and descriptors.

@File    :   landmark.py
@Time    :   2021/06/28 11:07:03
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import cv2
import numpy as np
import torch
from sklearn.preprocessing import normalize

from ..utils.base_model import BaseModel


class Landmark(BaseModel):
    """SIFT-based landmark extractor for keypoint detection and description."""
    
    default_conf = {
        'sift': False,
    }
    required_inputs = ['image']

    def _init(self, conf, model_path):
        """Initialize the landmark extractor with SIFT detector."""
        self.conf = {**self.default_conf, **conf}
        self.with_sift = self.conf['sift']
        
        if self.with_sift:
            self.sift = cv2.SIFT_create(nfeatures=self.conf['topk'])

    def _forward(self, data):
        """Extract SIFT keypoints and descriptors from input image.
        
        Args:
            data (dict): Input data containing 'image' tensor
            
        Returns:
            dict: Contains keypoints, scores, and descriptors lists
        """
        if self.with_sift:
            # Convert tensor to numpy format for OpenCV processing
            image = convert_tensor_to_numpy(data['image'])
            kpts, descs = self.sift.detectAndCompute(image, None)

            # Extract keypoint coordinates and response scores
            kpts_array = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
            scores_array = np.array([kp.response for kp in kpts])
            
            # L2 normalize descriptors
            descs = normalize(descs, norm='l2', axis=1)

            # Convert to tensors
            coord = torch.from_numpy(kpts_array).float()
            scores = torch.from_numpy(scores_array).float()
            descs = torch.from_numpy(descs).float().T if descs is not None else None

            return {
                'keypoints': [coord],
                'scores': [scores],
                'descriptors': [descs],
            }
        else:
            return


def convert_tensor_to_numpy(tensor):
    """Convert PyTorch tensor to OpenCV-compatible numpy array.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape BCHW
        
    Returns:
        np.ndarray: Numpy array in HWC format with uint8 dtype
        
    Raises:
        ValueError: If input tensor is not 4D (BCHW format)
    """
    # Ensure input tensor is 4D (BCHW)
    if len(tensor.shape) != 4:
        raise ValueError("Input tensor must be of shape BCHW")

    # Extract first image from batch
    tensor = tensor[0]

    # Convert from CHW to HWC for color images
    if tensor.shape[0] == 3:  # RGB image
        tensor = tensor.permute(1, 2, 0).contiguous()

    # Convert to numpy array
    numpy_image = tensor.cpu().numpy()

    # Ensure uint8 dtype and convert normalized [0,1] range to [0,255]
    if numpy_image.dtype != np.uint8:
        numpy_image = (numpy_image * 255).astype(np.uint8)

    return numpy_image

