import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from typing import Optional

# Try to import flash attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class RoPEPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create frequency tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len, device):
        """Generate RoPE encoding for given sequence length"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb
    
    def apply_rope(self, x, rope_cache):
        """Apply RoPE to input tensor x
        Args:
            x: (B_, num_heads, seq_len, head_dim) - query or key tensor
            rope_cache: (seq_len, head_dim) - precomputed rope encoding
        Returns:
            rotated_x: (B_, num_heads, seq_len, head_dim) - rotated tensor
        """
        seq_len = x.shape[-2]  
        head_dim = x.shape[-1]  
        rope = rope_cache[:seq_len]  # (seq_len, head_dim)
        
        # Expand rope to match x dimensions: (1, 1, seq_len, head_dim)
        rope = rope.unsqueeze(0).unsqueeze(0)
        cos_rope = rope.cos()
        sin_rope = rope.sin()
        
        # Split x along head_dim into two halves for RoPE rotation
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        cos_1, sin_1 = cos_rope[..., :head_dim//2], sin_rope[..., :head_dim//2]
        
        # Apply standard RoPE rotation
        rotated_x = torch.cat([
            x1 * cos_1 - x2 * sin_1,  # 实部
            x1 * sin_1 + x2 * cos_1   # 虚部
        ], dim=-1)
        
        return rotated_x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttentionWithRoPE(nn.Module):
    """Window based multi-head self attention with RoPE and multiple attention choices"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 attn_type='flash'):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.attn_type = attn_type

        # Validate attention type
        if attn_type == 'flash' and not FLASH_ATTN_AVAILABLE:
            print("Flash Attention not available, falling back to cosine attention")
            self.attn_type = 'cosine'

        # RoPE for position encoding
        self.rope = RoPEPositionalEncoding(head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # For cosine attention
        if self.attn_type == 'cosine':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, head_dim

        # Apply RoPE to q and k
        rope_cache = self.rope(N, device=x.device)
        q = self.rope.apply_rope(q, rope_cache)
        k = self.rope.apply_rope(k, rope_cache)

        if self.attn_type == 'flash' and mask is None:
            # Flash Attention 2.0 - most efficient for long sequences
            # Reshape for flash attention: (B_, N, num_heads, head_dim)
            q = q.transpose(1, 2)  # B_, N, num_heads, head_dim
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            x = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            x = x.reshape(B_, N, C)
            
        elif self.attn_type == 'cosine':
            # Cosine Attention - better for correlation tasks
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            # Fix device mismatch by ensuring tensor is on the same device as logit_scale
            max_logit = torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))
            logit_scale = torch.clamp(self.logit_scale, max=max_logit).exp()
            attn = attn * logit_scale
            
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            
        elif self.attn_type == 'linear':
            # Linear Attention - O(N) complexity
            q = F.elu(q) + 1
            k = F.elu(k) + 1
            
            # Compute linear attention
            kv = torch.einsum('bhnd,bhnf->bhdf', k, v)
            normalizer = torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2))
            x = torch.einsum('bhnd,bhdf->bhnf', q, kv) / (normalizer.unsqueeze(-1) + 1e-6)
            x = x.transpose(1, 2).reshape(B_, N, C)
            
        else:
            # Standard scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention specifically designed for dual-view correlation"""
    
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        # Separate projections for each modality
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Cross-modal mixing
        self.cross_attn_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        Args:
            x1: (B, N, C) - first modality
            x2: (B, N, C) - second modality  
        Returns:
            out1: (B, N, C) - enhanced first modality
            out2: (B, N, C) - enhanced second modality
        """
        B, N, C = x1.shape
        
        # Project to q, k, v
        q1 = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k1 = self.k_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v1 = self.v_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        q2 = self.q_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k2 = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v2 = self.v_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Self-attention
        attn11 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn22 = (q2 @ k2.transpose(-2, -1)) * self.scale
        
        # Cross-attention  
        attn12 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn21 = (q2 @ k1.transpose(-2, -1)) * self.scale
        
        # Mix self and cross attention
        attn1 = (1 - self.cross_attn_scale) * attn11 + self.cross_attn_scale * attn12
        attn2 = (1 - self.cross_attn_scale) * attn22 + self.cross_attn_scale * attn21
        
        attn1 = self.softmax(attn1)
        attn2 = self.softmax(attn2)
        
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)
        
        # Apply attention
        out1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        out2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
        
        out1 = self.proj(out1)
        out2 = self.proj(out2)
        out1 = self.proj_drop(out1)
        out2 = self.proj_drop(out2)
        
        return out1, out2


class DoubleBlock(nn.Module):
    """Double Block - processes concatenated features from both views"""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='cosine'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window-size"

        # Process concatenated features (2*dim)
        self.norm1 = norm_layer(dim * 2)
        self.attn = WindowAttentionWithRoPE(
            dim * 2, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, attn_type=attn_type)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim * 2)
        mlp_hidden_dim = int(dim * 2 * mlp_ratio)
        self.mlp = Mlp(in_features=dim * 2, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Attention mask for shifted window
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x0, x1):
        """
        Args:
            x0: (B, L, C) - first view
            x1: (B, L, C) - second view
        Returns:
            x_concat: (B, L, 2*C) - concatenated and processed features
        """
        H, W = self.input_resolution
        B, L, C = x0.shape
        assert L == H * W, "input feature has wrong size"

        # Concatenate along channel dimension (dim=-1)
        x_concat = torch.cat([x0, x1], dim=-1)  # (B, L, 2*C) ✓ 正确：沿channel维度拼接
        
        shortcut = x_concat
        x_concat = self.norm1(x_concat)
        x_concat = x_concat.view(B, H, W, 2*C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_concat, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_concat

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, 2*C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, 2*C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_concat = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_concat = shifted_x
        x_concat = x_concat.view(B, H * W, 2*C)

        # FFN
        x_concat = shortcut + self.drop_path(x_concat)
        x_concat = x_concat + self.drop_path(self.mlp(self.norm2(x_concat)))

        return x_concat


class SingleBlock(nn.Module):
    """Single Block - processes each view independently after splitting"""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='flash'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window-size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionWithRoPE(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, attn_type=attn_type)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Attention mask for shifted window
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """Process single view with self-attention and RoPE"""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA with RoPE
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerV2_CCT(nn.Module):
    r""" Swin Transformer V2 with D×N + S×M architecture for dual input processing.
    
    Args:
        feat_size (int): Input feature size. Default 40
        feat_chan (int): Number of input feature channels. Default: 256
        double_depths (tuple(int)): Depth of each Double layer (D×N).
        single_depths (tuple(int): Depth of each Single layer (S×M).
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, feat_size=40, feat_chan=256, double_depths=[2, 2], single_depths=[6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 double_attn_type='cosine', single_attn_type='flash'):
        super().__init__()

        self.feat_size = feat_size
        self.feat_chan = feat_chan
        self.flat_len = feat_size * feat_size
        
        # Total layers: D×N + S×M
        total_double_blocks = sum(double_depths)
        total_single_blocks = sum(single_depths)
        total_blocks = total_double_blocks + total_single_blocks

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Build Double layers (D×N)
        self.double_layers = nn.ModuleList()
        block_idx = 0
        for i, depth in enumerate(double_depths):
            layer_blocks = nn.ModuleList()
            for j in range(depth):
                block = DoubleBlock(
                    dim=feat_chan,
                    input_resolution=(feat_size, feat_size),
                    num_heads=num_heads[min(i, len(num_heads)-1)],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    norm_layer=norm_layer,
                    attn_type=double_attn_type)
                layer_blocks.append(block)
                block_idx += 1
            self.double_layers.append(layer_blocks)

        # Build Single layers (S×M)
        self.single_layers = nn.ModuleList()
        for i, depth in enumerate(single_depths):
            layer_blocks = nn.ModuleList()
            for j in range(depth):
                block = SingleBlock(
                    dim=feat_chan,
                    input_resolution=(feat_size, feat_size),
                    num_heads=num_heads[min(len(double_depths) + i, len(num_heads)-1)],
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    norm_layer=norm_layer,
                    attn_type=single_attn_type)
                layer_blocks.append(block)
                block_idx += 1
            self.single_layers.append(layer_blocks)

        self.apply(_init_weights)

    def forward(self, x0, x1):
        """
        Args:
            x0: (B, L, C) where L = H * W - first view
            x1: (B, L, C) where L = H * W - second view
        Returns:
            x0: (B, L, C) - processed first view
            x1: (B, L, C) - processed second view
        """
        # Ensure input shapes are preserved
        B, L, C = x0.shape
        assert x1.shape == (B, L, C), f"Input shapes must match: x0={x0.shape}, x1={x1.shape}"
        assert L == self.flat_len, f"Input length {L} must match expected {self.flat_len}"
        assert C == self.feat_chan, f"Input channels {C} must match expected {self.feat_chan}"

        # Phase 1: Double layers (D×N) - concatenated processing
        x_concat = None
        for layer_blocks in self.double_layers:
            if x_concat is None:
                # First double layer: concatenate inputs and process
                for block in layer_blocks:
                    x_concat = block(x0, x1)  # x_concat shape: (B, L, 2*C)
            else:
                # Subsequent double layers: process concatenated features
                for block in layer_blocks:
                    # Split along channel dimension (dim=-1) to get original C channels each
                    x0_temp, x1_temp = torch.chunk(x_concat, 2, dim=-1)  # ✓ 正确：在channel维度split
                    # x0_temp: (B, L, C), x1_temp: (B, L, C)
                    x_concat = block(x0_temp, x1_temp)  # 重新拼接得到 (B, L, 2*C)

        # Phase 2: Single layers (S×M) - independent processing
        if x_concat is not None:
            # Split concatenated features back to individual views
            x0, x1 = torch.chunk(x_concat, 2, dim=-1)  # ✓ 正确：在channel维度split
            # x0: (B, L, C), x1: (B, L, C)
        
        for layer_blocks in self.single_layers:
            for block in layer_blocks:
                x0 = block(x0)  # 独立处理每个view
                x1 = block(x1)

        # Ensure output shapes are preserved
        assert x0.shape == (B, L, C), f"Output x0 shape changed: expected {(B, L, C)}, got {x0.shape}"
        assert x1.shape == (B, L, C), f"Output x1 shape changed: expected {(B, L, C)}, got {x1.shape}"
        
        return x0, x1
