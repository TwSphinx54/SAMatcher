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
    """Rotary Position Embedding for enhanced position awareness in attention"""
    
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
        """Apply RoPE rotation to query/key tensors
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


class WindowAttentionWithRoPE(nn.Module):
    """Window-based multi-head attention with RoPE and multiple attention variants"""

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

        # Fallback if flash attention unavailable
        if attn_type == 'flash' and not FLASH_ATTN_AVAILABLE:
            print("Flash Attention not available, falling back to cosine attention")
            self.attn_type = 'cosine'

        # RoPE for position encoding
        self.rope = RoPEPositionalEncoding(head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Learnable scale for cosine attention
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
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                x = flash_attn_func(q.half(), k.half(), v.half(), 
                                   dropout_p=self.attn_drop.p if self.training else 0.0)
            x = x.float().reshape(B_, N, C)
            
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


class DoubleBlock(nn.Module):
    """Processes concatenated features from both views for cross-view interaction"""
    
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

    def forward(self, x0, x1):
        """Concatenate and process both views together"""
        H, W = self.input_resolution
        B, L, C = x0.shape
        assert L == H * W, "input feature has wrong size"

        # Concatenate along channel dimension
        x_concat = torch.cat([x0, x1], dim=-1)  # (B, L, 2*C) ✓ 正确：沿channel维度拼接
        
        shortcut = x_concat
        x_concat = self.norm1(x_concat)
        x_concat = x_concat.view(B, H, W, 2*C)

        # Shifted window attention
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

        # Residual connection and MLP
        x_concat = shortcut + self.drop_path(x_concat)
        x_concat = x_concat + self.drop_path(self.mlp(self.norm2(x_concat)))

        return x_concat


class SingleBlock(nn.Module):
    """Processes each view independently for view-specific features"""
    
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

    def forward(self, x):
        """Process single view with self-attention"""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

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

        # Residual connection and MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformerV2_CCT(nn.Module):
    """Swin Transformer V2 with D×N + S×M architecture for dual-view processing
    
    Architecture: D×N Double blocks for cross-view interaction + S×M Single blocks for view-specific processing
    """

    def __init__(self, feat_size=40, feat_chan=256, double_depths=[2, 2], single_depths=[6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 double_attn_type='cosine', single_attn_type='flash'):
        super().__init__()

        self.feat_size = feat_size
        self.feat_chan = feat_chan
        self.flat_len = feat_size * feat_size
        
        # Calculate total blocks for stochastic depth scheduling
        total_double_blocks = sum(double_depths)
        total_single_blocks = sum(single_depths)
        total_blocks = total_double_blocks + total_single_blocks

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Build Double layers for cross-view interaction
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

        # Build Single layers for view-specific processing
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
        """Forward pass through D×N + S×M architecture"""
        B, L, C = x0.shape
        assert x1.shape == (B, L, C), f"Input shapes must match: x0={x0.shape}, x1={x1.shape}"
        assert L == self.flat_len, f"Input length {L} must match expected {self.flat_len}"
        assert C == self.feat_chan, f"Input channels {C} must match expected {self.feat_chan}"

        # Phase 1: Double layers - cross-view interaction
        x_concat = None
        for layer_blocks in self.double_layers:
            if x_concat is None:
                # First double layer: concatenate inputs
                for block in layer_blocks:
                    x_concat = block(x0, x1)
            else:
                # Subsequent layers: process concatenated features
                for block in layer_blocks:
                    x0_temp, x1_temp = torch.chunk(x_concat, 2, dim=-1)
                    x_concat = block(x0_temp, x1_temp)

        # Phase 2: Single layers - view-specific processing
        if x_concat is not None:
            # Split concatenated features back to individual views
            x0, x1 = torch.chunk(x_concat, 2, dim=-1)
        
        for layer_blocks in self.single_layers:
            for block in layer_blocks:
                x0 = block(x0)
                x1 = block(x1)

        return x0, x1
