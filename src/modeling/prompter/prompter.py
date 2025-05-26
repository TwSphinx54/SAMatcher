import torch
from einops import rearrange
import torch.nn as nn
from src.modeling.sam2_utils import generate_mesh_grid
from src.modeling.prompter.transformer import TransformerDecoder


class Prompter(TransformerDecoder):
    def __init__(self, d_model, nhead, feat_size, no_ker_size, num_layers):
        super().__init__(d_model, nhead, feat_size, no_ker_size, num_layers)
        
        self.query = nn.Embedding(1, d_model)
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                (3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 1, (1, 1)),
        )
        self.softmax_temperature = 1

    def forward(self, emb0, emb1, h_emb, w_emb, h, w):
        query_weight = self.query.weight.unsqueeze(0).expand(emb0.shape[0], -1, -1)
        prompt_query0 = query_weight.clone()
        prompt_query1 = query_weight.clone()
    
        prompt_query0 = super().forward(1, prompt_query0, emb0)
        prompt_query1 = super().forward(1, prompt_query1, emb1)

        ap0, ap1, labels = self.ap_estimation(prompt_query0, emb0, prompt_query1, emb1, h_emb, w_emb, h, w)

        return ap0, ap1, labels

    def ap_estimation(self, prompt_query0, image_embedding0, prompt_query1, image_embedding1, h, w, H, W):
        N, num_points = prompt_query0.shape[:2]
        att0 = torch.einsum('blc, bnc->bln', image_embedding0, prompt_query0)  # [N, hw, num_q]
        att1 = torch.einsum('blc, bnc->bln', image_embedding1, prompt_query1)  # [N, hw, num_q]

        # Adjust dimensions for element-wise multiplication
        att0 = att0.unsqueeze(2)  # [N, hw, 1, num_q]
        image_embedding0 = image_embedding0.unsqueeze(-1)  # [N, hw, c, 1]
        att1 = att1.unsqueeze(2)  # [N, hw, 1, num_q]
        image_embedding1 = image_embedding1.unsqueeze(-1)  # [N, hw, c, 1]

        # weighted sum for center regression
        heatmap0 = rearrange(image_embedding0 * att0, 'n (h w) c q -> n q c h w', h=h, w=w)
        heatmap1 = rearrange(image_embedding1 * att1, 'n (h w) c q -> n q c h w', h=h, w=w)

        # Merge q dimension with batch dimension n
        heatmap0 = heatmap0.reshape(-1, heatmap0.shape[2], heatmap0.shape[3], heatmap0.shape[4])  # [N*q, c, h, w]
        heatmap1 = heatmap1.reshape(-1, heatmap1.shape[2], heatmap1.shape[3], heatmap1.shape[4])  # [N*q, c, h, w]

        # Apply heatmap_conv
        heatmap_flatten0 = rearrange(
            self.heatmap_conv(heatmap0),
            '(n q) c h w -> n q (h w) c', q=num_points
        ) * self.softmax_temperature  # [N, q, h*w, c]
        heatmap_flatten1 = rearrange(
            self.heatmap_conv(heatmap1),
            '(n q) c h w -> n q (h w) c', q=num_points
        ) * self.softmax_temperature  # [N, q, h*w, c]

        prob_map0 = nn.functional.softmax(heatmap_flatten0, dim=2)  # [N, q, h*w, c]
        prob_map1 = nn.functional.softmax(heatmap_flatten1, dim=2)  # [N, q, h*w, c]

        stride_h = H // h
        stride_w = W // w
        coord_xy_map0 = generate_mesh_grid(
            (h, w),
            stride_h=stride_h,
            stride_w=stride_w,
            device=image_embedding0.device
        ).unsqueeze(1)  # [1, 1, h*w, 2]
        coord_xy_map1 = generate_mesh_grid(
            (h, w),
            stride_h=stride_h,
            stride_w=stride_w,
            device=image_embedding1.device
        ).unsqueeze(1)  # [1, 1, h*w, 2]

        ap0 = (prob_map0 * coord_xy_map0).sum(2)  # [N, num_q, 2]
        ap1 = (prob_map1 * coord_xy_map1).sum(2)  # [N, num_q, 2]

        # Create labels for both images (each image has num_points points)
        labels = torch.ones((N, num_points * 2), dtype=torch.int64, device=image_embedding0.device)
        return ap0, ap1, labels
