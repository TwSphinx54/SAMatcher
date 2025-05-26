import torch
import torch.nn as nn

class LWAPP(nn.Module):
    """
    平衡计算效率和学习能力的语义中心匹配模块
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 降维但保持一定的表达能力
        self.feature_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # 轻量级交叉注意力：使用低秩分解
        self.q_proj = nn.Linear(d_model // 4, d_model // 8)
        self.k_proj = nn.Linear(d_model // 4, d_model // 8)
        
        # 空间注意力权重
        self.spatial_weight = nn.Parameter(torch.randn(d_model // 4) * 0.01)
        
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, emb0, emb1, h_emb, w_emb, H, W):
        """
        Args:
            emb0: [B, N, C] - First image embeddings
            emb1: [B, N, C] - Second image embeddings  
            h_emb, w_emb: Height and width of embedding feature map
            H, W: Original image height and width
            
        Returns:
            points0: [B, 1, 2] - Predicted points for image 0
            points1: [B, 1, 2] - Predicted points for image 1  
            labels: [B, 2] - Labels for both points (all ones)
        """
        B, N, C = emb0.shape
        
        # 1. 特征投影降维
        proj_emb0 = self.feature_proj(emb0)  # [B, N, C//4]
        proj_emb1 = self.feature_proj(emb1)  # [B, N, C//4]
        
        # 2. 计算空间注意力权重
        spatial_scores0 = torch.sum(proj_emb0 * self.spatial_weight, dim=-1)  # [B, N]
        spatial_scores1 = torch.sum(proj_emb1 * self.spatial_weight, dim=-1)  # [B, N]
        
        # 3. 轻量级交叉匹配：使用低秩attention
        q0 = self.q_proj(proj_emb0)  # [B, N, C//8]
        k1 = self.k_proj(proj_emb1)  # [B, N, C//8]
        
        q1 = self.q_proj(proj_emb1)  # [B, N, C//8]
        k0 = self.k_proj(proj_emb0)  # [B, N, C//8]
        
        # 计算交叉相似度 (使用矩阵乘法但维度更小)
        cross_sim_0_to_1 = torch.bmm(q0, k1.transpose(1, 2))  # [B, N, N]
        cross_sim_1_to_0 = torch.bmm(q1, k0.transpose(1, 2))  # [B, N, N]
        
        # 4. 结合空间权重和交叉相似度
        # 对于每个位置，找到最匹配的对应位置
        max_cross_sim_0, max_indices_0 = torch.max(cross_sim_0_to_1, dim=2)  # [B, N]
        max_cross_sim_1, max_indices_1 = torch.max(cross_sim_1_to_0, dim=2)  # [B, N]
        
        # 综合评分：空间重要性 + 交叉匹配度
        combined_scores_0 = spatial_scores0 + max_cross_sim_0 * 0.5
        combined_scores_1 = spatial_scores1 + max_cross_sim_1 * 0.5
        
        # 5. 选择最优位置
        best_idx_0 = torch.argmax(combined_scores_0, dim=1)  # [B]
        best_idx_1 = torch.argmax(combined_scores_1, dim=1)  # [B]
        
        # 6. 转换为坐标
        points0 = self._idx_to_coords(best_idx_0, h_emb, w_emb, H, W).unsqueeze(1)  # [B, 1, 2]
        points1 = self._idx_to_coords(best_idx_1, h_emb, w_emb, H, W).unsqueeze(1)  # [B, 1, 2]
        
        # 7. 创建labels - 两个点都是正样本
        labels = torch.ones((B, 2), dtype=torch.int64, device=emb0.device)  # [B, 2]
        
        return points0, points1, labels
        
    def _idx_to_coords(self, indices, h_emb, w_emb, H, W):
        """Convert flattened indices to (x, y) coordinates."""
        row = indices // w_emb
        col = indices % w_emb
        
        stride_h = H / h_emb
        stride_w = W / w_emb
        
        y = (row.float() + 0.5) * stride_h
        x = (col.float() + 0.5) * stride_w
        
        return torch.stack([x, y], dim=-1)  # [B, 2]