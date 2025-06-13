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
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition to original spatial layout"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowCrossAttention(nn.Module):
    """Window-based cross-attention for correlation between two views"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Separate projections for query (from x1) and key-value (from x0)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x0, x1, mask: Optional[torch.Tensor] = None):
        """
        Cross attention: x1 queries x0
        Args:
            x0: key-value features (num_windows*B, N, C)
            x1: query features (num_windows*B, N, C)  
            mask: attention mask
        """
        B_, N, C = x1.shape
        
        # Generate query from x1, key-value from x0
        q = self.q_proj(x1).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(x0).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Scaled dot-product attention with relative position bias
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

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


class WindowSelfAttention(nn.Module):
    """Standard window-based self-attention"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """Standard self-attention"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

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


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with configurable attention type"""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='cross'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.attn_type = attn_type  # 'cross' or 'self'
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if attn_type == 'cross':
            self.norm1_x0 = norm_layer(dim)
            self.norm1_x1 = norm_layer(dim)
            self.attn = WindowCrossAttention(
                dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:  # self attention
            self.norm1 = norm_layer(dim)
            self.attn = WindowSelfAttention(
                dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Create attention mask for shifted windows
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

    def forward(self, x0, x1=None):
        """
        Forward pass with different attention types
        Args:
            x0: first feature map
            x1: second feature map (required for cross-attention)
        """
        H, W = self.input_resolution
        B, L, C = x0.shape
        assert L == H * W, "input feature has wrong size"
        
        if self.attn_type == 'cross':
            assert x1 is not None and x1.shape == x0.shape, "x1 required for cross attention"
            return self._forward_cross(x0, x1)
        else:
            return self._forward_self(x0)

    def _forward_cross(self, x0, x1):
        """Cross attention between x0 and x1"""
        H, W = self.input_resolution
        B, L, C = x0.shape

        shortcut = x1
        x0 = self.norm1_x0(x0)
        x1 = self.norm1_x1(x1)
        x0 = x0.view(B, H, W, C)
        x1 = x1.view(B, H, W, C)

        # Shifted window attention
        if self.shift_size > 0:
            shifted_x0 = torch.roll(x0, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x0 = x0
            shifted_x1 = x1

        # Partition windows
        x0_windows = window_partition(shifted_x0, self.window_size)
        x1_windows = window_partition(shifted_x1, self.window_size)
        x0_windows = x0_windows.view(-1, self.window_size * self.window_size, C)
        x1_windows = x1_windows.view(-1, self.window_size * self.window_size, C)

        # Cross attention: x1 queries x0
        attn_windows = self.attn(x0_windows, x1_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Residual connection and MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def _forward_self(self, x):
        """Self attention on single feature map"""
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Self attention
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

        # Residual connection and MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Swin2Correlator(nn.Module):
    """Swin Transformer V2 for correlation between two feature maps
    
    Supports alternating self-attention and cross-attention patterns.
    Common patterns:
    - 'cross': Pure cross-attention (like RAFT correlation)
    - 'alternate': Alternating self -> cross -> self -> cross...
    - 'block_alternate': Block-wise alternating (self block -> cross block)
    """

    def __init__(self, feat_size=40, feat_chan=256, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 correlation_mode='alternate'):
        super().__init__()

        self.feat_size = feat_size
        self.feat_chan = feat_chan
        self.flat_len = feat_size * feat_size
        self.num_layers = len(depths)
        self.correlation_mode = correlation_mode

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers with different attention patterns
        self.layers = nn.ModuleList()
        block_idx = 0
        
        for i_layer in range(self.num_layers):
            layer_blocks = nn.ModuleList()
            for i in range(depths[i_layer]):
                # Determine attention type based on mode
                if correlation_mode == 'cross':
                    attn_type = 'cross'
                elif correlation_mode == 'alternate':
                    attn_type = 'cross' if block_idx % 2 == 1 else 'self'
                elif correlation_mode == 'block_alternate':
                    attn_type = 'cross' if i_layer % 2 == 1 else 'self'
                else:
                    attn_type = 'self'
                
                block = SwinTransformerBlock(
                    dim=feat_chan,
                    input_resolution=(feat_size, feat_size),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    norm_layer=norm_layer,
                    attn_type=attn_type)
                layer_blocks.append(block)
                block_idx += 1
            self.layers.append(layer_blocks)

        self.norm = norm_layer(feat_chan)
        self.apply(_init_weights)

    def forward(self, x0, x1):
        """
        Correlation between two feature maps
        Args:
            x0: (B, L, C) - reference features
            x1: (B, L, C) - query features
        Returns:
            x0_out, x1_out: (B, L, C) - processed features from both views
        """
        B, L, C = x0.shape
        assert x1.shape == (B, L, C), f"Input shapes must match: x0={x0.shape}, x1={x1.shape}"
        assert L == self.flat_len, f"Input length {L} must match expected {self.flat_len}"
        assert C == self.feat_chan, f"Input channels {C} must match expected {self.feat_chan}"

        # Process through transformer layers
        for layer_blocks in self.layers:
            for block in layer_blocks:
                if block.attn_type == 'cross':
                    # Cross attention: both views interact
                    x1_new = block(x0, x1)  # x1 queries x0
                    x0_new = block(x1, x0)  # x0 queries x1 (bidirectional)
                    x0, x1 = x0_new, x1_new
                else:
                    # Self attention: process each view independently
                    x0 = block(x0)
                    x1 = block(x1)

        x0 = self.norm(x0)
        x1 = self.norm(x1)
        return x0, x1
