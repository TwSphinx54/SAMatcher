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
    """Initialize weights for linear and layer norm layers"""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class ViewAwareRoPE(nn.Module):
    """Rotary Position Embedding with view-aware capabilities for enhanced position encoding"""
    
    def __init__(self, dim, max_seq_len=10000, view_offset=0.1):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.view_offset = view_offset
        
        # Create frequency tensor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward_interleaved(self, seq_len, device, view_aware=False):
        """Generate RoPE encoding for interleaved features with optional view distinction
        
        Args:
            seq_len: sequence length for interleaved features (2 * original_length)
            device: torch device
            view_aware: if True, add slight offset to distinguish views
        """
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        
        if view_aware:
            # Add slight offset for odd positions (view1) to distinguish from even positions (view0)
            view_offsets = torch.zeros_like(t)
            view_offsets[1::2] = self.view_offset
            t = t + view_offsets
            
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb
    
    def forward_single_view(self, seq_len, device, view_id=0, view_aware=False):
        """Generate RoPE encoding for single view with optional view distinction
        
        Args:
            seq_len: sequence length for single view
            device: torch device  
            view_id: 0 for first view, 1 for second view
            view_aware: if True, add view-specific offset
        """
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        
        if view_aware:
            # Add view-specific offset
            t = t + view_id * self.view_offset
            
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
            x1 * cos_1 - x2 * sin_1,  # Real part
            x1 * sin_1 + x2 * cos_1   # Imaginary part
        ], dim=-1)
        
        return rotated_x


class Mlp(nn.Module):
    """Multi-layer Perceptron with GELU activation"""
    
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
    """Partition into non-overlapping windows
    
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition to original spatial layout
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: window size
        H: height of image
        W: width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttentionWithViewAwareRoPE(nn.Module):
    """Window-based attention with configurable view-aware RoPE for enhanced position encoding"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 attn_type='flash', view_aware=False, view_offset=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.attn_type = attn_type
        self.view_aware = view_aware

        # Fallback if flash attention unavailable
        if attn_type == 'flash' and not FLASH_ATTN_AVAILABLE:
            print("Flash Attention not available, falling back to cosine attention")
            self.attn_type = 'cosine'

        # View-aware RoPE for position encoding
        self.rope = ViewAwareRoPE(head_dim, view_offset=view_offset)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Learnable scale for cosine attention
        if self.attn_type == 'cosine':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        
        self.softmax = nn.Softmax(dim=-1)

        # Flash Attention compatibility detection
        self._flash_attn_available = FLASH_ATTN_AVAILABLE

    def _prepare_flash_attn_inputs(self, q, k, v):
        """Prepare inputs for Flash Attention with proper dtype conversion"""
        # Check if any tensor is already in supported dtype
        supported_dtypes = [torch.float16, torch.bfloat16]
        current_dtype = q.dtype
        
        # If already in supported dtype, return as is
        if current_dtype in supported_dtypes:
            return q, k, v, current_dtype
        
        # Convert to fp16 for Flash Attention (most widely supported)
        target_dtype = torch.float16
        
        # Check if the device supports fp16
        if q.device.type == 'cuda':
            try:
                # Test if fp16 is supported on this device
                test_tensor = torch.ones(1, device=q.device, dtype=torch.float16)
                target_dtype = torch.float16
            except RuntimeError:
                # Fallback to bf16 if fp16 not supported
                target_dtype = torch.bfloat16
        else:
            # For CPU, bf16 is usually better supported
            target_dtype = torch.bfloat16
        
        q_converted = q.to(target_dtype)
        k_converted = k.to(target_dtype)
        v_converted = v.to(target_dtype)
        
        return q_converted, k_converted, v_converted, target_dtype

    def forward(self, x, mask: Optional[torch.Tensor] = None, is_interleaved=False, view_id=0):
        """Window attention forward pass with view-aware RoPE"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, num_heads, N, head_dim

        # Apply view-aware RoPE to q and k
        if is_interleaved:
            # For interleaved features (DoubleBlock)
            rope_cache = self.rope.forward_interleaved(N, device=x.device, view_aware=self.view_aware)
        else:
            # For single view features (SingleBlock)  
            rope_cache = self.rope.forward_single_view(N, device=x.device, view_id=view_id, view_aware=self.view_aware)
            
        q = self.rope.apply_rope(q, rope_cache)
        k = self.rope.apply_rope(k, rope_cache)

        # Attention computation based on type
        if self.attn_type == 'flash' and mask is None and self._flash_attn_available:
            # Prepare inputs for Flash Attention with proper dtype
            q_flash, k_flash, v_flash, flash_dtype = self._prepare_flash_attn_inputs(q, k, v)
            
            # Transpose for Flash Attention: (B_, num_heads, N, head_dim) -> (B_, N, num_heads, head_dim)
            q_flash = q_flash.transpose(1, 2)
            k_flash = k_flash.transpose(1, 2)
            v_flash = v_flash.transpose(1, 2)
            
            try:
                # Use Flash Attention with converted dtype
                out = flash_attn_func(q_flash, k_flash, v_flash, 
                                     dropout_p=self.attn_drop.p if self.training else 0.0)
                # Convert back to original dtype if needed
                if flash_dtype != x.dtype:
                    out = out.to(x.dtype)
                x = out.reshape(B_, N, C)
                
            except Exception as e:
                # Fallback to standard attention
                import warnings
                warnings.warn(
                    f"Flash Attention failed: {e}. Falling back to standard attention.",
                    category=UserWarning,
                    stacklevel=2,
                )
                # Use original tensors for fallback (not converted ones)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = self.softmax(attn)
                attn = self.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            
        elif self.attn_type == 'cosine':
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
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
            q = F.elu(q) + 1
            k = F.elu(k) + 1
            
            kv = torch.einsum('bhnd,bhnf->bhdf', k, v)
            normalizer = torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2))
            x = torch.einsum('bhnd,bhdf->bhnf', q, kv) / (normalizer.unsqueeze(-1) + 1e-6)
            x = x.transpose(1, 2).reshape(B_, N, C)
            
        else:  # Standard scaled dot-product attention
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
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='cosine',
                 view_aware=False, view_offset=0.1):
        super().__init__()
        self.dim = dim
        self.input_resolution = (input_resolution[0], input_resolution[1] * 2)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.view_aware = view_aware
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window-size"

        # Process concatenated features with view-aware RoPE
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionWithViewAwareRoPE(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, attn_type=attn_type,
            view_aware=view_aware, view_offset=view_offset)

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

    def forward(self, x0, x1):
        """Ensure windows contain mixed features from both views for cross-view interaction"""
        H, W = self.input_resolution
        original_W = W // 2
        B, L, C = x0.shape
        
        # Interleave features spatially
        x0_spatial = x0.view(B, H, original_W, C)
        x1_spatial = x1.view(B, H, original_W, C)
        
        # Interleave along width: [x0_col0, x1_col0, x0_col1, x1_col1, ...]
        x_interleaved = torch.stack([x0_spatial, x1_spatial], dim=3)  # (B, H, original_W, 2, C)
        x_concat = x_interleaved.view(B, H, W, C)  # (B, H, 2*original_W, C)
        
        # Correct shortcut dimensions
        shortcut = x_concat.view(B, H * W, C)  # (B, H * 2*original_W, C)
        x_concat_flat = self.norm1(shortcut)
        x_concat = x_concat_flat.view(B, H, W, C)
        
        # Shifted window attention
        if self.shift_size > 0:
            shifted_x = torch.roll(x_concat, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_concat

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA with view-aware RoPE for interleaved features
        attn_windows = self.attn(x_windows, mask=self.attn_mask, is_interleaved=True)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_concat = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_concat = shifted_x
        x_concat = x_concat.view(B, H * W, C)

        # Residual connection and MLP
        x_concat = shortcut + self.drop_path(x_concat)
        x_concat = x_concat + self.drop_path(self.mlp(self.norm2(x_concat)))

        return x_concat

    def split_interleaved_features(self, x_concat):
        """Split interleaved features back to separate views"""
        H, W = self.input_resolution
        original_W = W // 2
        B, _, C = x_concat.shape
        
        x_spatial = x_concat.view(B, H, W, C)  # (B, H, 2*original_W, C)
        
        # Split back to interleaved format
        x_interleaved = x_spatial.view(B, H, original_W, 2, C)  # (B, H, original_W, 2, C)
        
        # Extract two views
        x0_spatial = x_interleaved[:, :, :, 0, :]  # (B, H, original_W, C)
        x1_spatial = x_interleaved[:, :, :, 1, :]  # (B, H, original_W, C)
        
        # Flatten back to sequence format
        x0 = x0_spatial.view(B, H * original_W, C)
        x1 = x1_spatial.view(B, H * original_W, C)
        
        return x0, x1


class SingleBlock(nn.Module):
    """Processes each view independently for view-specific feature extraction"""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_type='flash',
                 view_aware=False, view_offset=0.1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.view_aware = view_aware
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window-size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionWithViewAwareRoPE(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, attn_type=attn_type,
            view_aware=view_aware, view_offset=view_offset)

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

    def forward(self, x, view_id=0):
        """Process single view with view-aware self-attention"""
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

        # W-MSA/SW-MSA with view-aware RoPE for single view
        attn_windows = self.attn(x_windows, mask=self.attn_mask, is_interleaved=False, view_id=view_id)

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


class SymSwinFuser(nn.Module):
    """Symmetric Swin Transformer with configurable view-aware RoPE for dual-view feature fusion"""

    def __init__(self, feat_size=40, feat_chan=256, double_depths=[2, 2], single_depths=[6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 double_attn_type='cosine', single_attn_type='flash',
                 view_aware=False, view_offset=0.1):
        super().__init__()

        self.feat_size = feat_size
        self.feat_chan = feat_chan
        self.flat_len = feat_size * feat_size
        self.view_aware = view_aware
        
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
                    attn_type=double_attn_type,
                    view_aware=view_aware,
                    view_offset=view_offset)
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
                    attn_type=single_attn_type,
                    view_aware=view_aware,
                    view_offset=view_offset)
                layer_blocks.append(block)
                block_idx += 1
            self.single_layers.append(layer_blocks)

        self.apply(_init_weights)

    def forward(self, x0, x1):
        """Forward pass through D×N + S×M architecture for dual-view feature fusion"""
        B, L, C = x0.shape
        assert x1.shape == (B, L, C), f"Input shapes must match: x0={x0.shape}, x1={x1.shape}"
        assert L == self.flat_len, f"Input length {L} must match expected {self.flat_len}"
        assert C == self.feat_chan, f"Input channels {C} must match expected {self.feat_chan}"

        # Phase 1: Double layers - cross-view interaction
        for i, layer_blocks in enumerate(self.double_layers):
            if i == 0:
                for block in layer_blocks:
                    x_concat = block(x0, x1)
            else:
                for block in layer_blocks:
                    x0_temp, x1_temp = block.split_interleaved_features(x_concat)
                    x_concat = block(x0_temp, x1_temp)

        # Phase 2: Single layers - view-specific processing
        if len(self.double_layers) > 0:
            last_double_block = self.double_layers[-1][-1]
            x0, x1 = last_double_block.split_interleaved_features(x_concat)
        
        for layer_blocks in self.single_layers:
            for block in layer_blocks:
                x0 = block(x0, view_id=0)  # Process first view
                x1 = block(x1, view_id=1)  # Process second view

        return x0, x1
