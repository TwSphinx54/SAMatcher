import math
import torch
import torch.nn as nn

from timm.layers import DropPath, trunc_normal_


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class DWConv(nn.Module):
    def __init__(self, dim=256):
        super(DWConv, self).__init__()
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwc(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwc = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwc(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CCAttention(nn.Module):
    r""" Cross-View Cross-Scale Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        proj_drop (float, optional): Dropout rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        no_ker_size (list): Non-overlap kernel sizes in Multi-Scale Aggregation Layer
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads=8,
                 attn_drop=0.,
                 proj_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 qkv_bias=False,
                 qk_scale=None,
                 no_ker_size=[2, 4, 8]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.input_resolution = input_resolution
        self.no_ker_size = no_ker_size
        self.msa = MSA(input_resolution=input_resolution, dim=dim, norm_layer=norm_layer, msa_size=no_ker_size,
                       stride=no_ker_size, split=True)

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q0 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q12 = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)
        self.kv0 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim // 2, dim // 2, bias=qkv_bias)

        self.act = act_layer()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim // 2)
        self.norm2 = nn.LayerNorm(dim // 2)

        self.dwc0 = DWConv(dim // 2)
        self.dwc1 = DWConv(dim // 4)
        self.dwc2 = DWConv(dim // 4)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(_init_weights)

    def attention(self, q, k, v, dwc, B, C, H, W):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        heads = self.num_heads // 2
        v = v + dwc(v.permute(0, 2, 1, 3).flatten(2), H, W).reshape(B, heads, C // heads, -1).permute(0, 1, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        return x

    def forward(self, x0, x1):
        H, W = self.input_resolution
        B, L, C = x0.shape
        x1s = self.msa(x1)  # B,256,H0,W0  B,128,H1,W1  B,128,H2,W2

        # B,H*W,8,32 → B,8,H*W,32
        q = self.q0(x0).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q0 = q[:, :self.num_heads // 2]  # B,4,H*W,32
        q12 = q[:, self.num_heads // 2:].permute(0, 2, 1, 3).flatten(2)  # B,H*W,128
        q12 = self.q12(q12).reshape(B, -1, self.num_heads, (C // 2) // self.num_heads).permute(0, 2, 1, 3)  # B,8,H*W,16
        q1, q2 = q12[:, :self.num_heads // 2], q12[:, self.num_heads // 2:]  # B,4,H*W,16

        x_0 = self.act(self.norm0(x1s[0].flatten(2).permute(0, 2, 1)))
        x_1 = self.act(self.norm1(x1s[1].flatten(2).permute(0, 2, 1)))
        x_2 = self.act(self.norm2(x1s[2].flatten(2).permute(0, 2, 1)))

        # B,H0*W0,2,4,32 → 2,B,4,H0*W0,32
        kv0 = self.kv0(x_0).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B,H1*W1,2,4,16 → 2,B,4,H1*W1,16
        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, (C // 2) // self.num_heads).permute(2, 0, 3, 1, 4)
        # B,H2*W2,2,4,16 → 2,B,4,H2*W2,16
        kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, (C // 2) // self.num_heads).permute(2, 0, 3, 1, 4)

        k0, v0 = kv0[0], kv0[1]  # B,4,H0*W0,32
        k1, v1 = kv1[0], kv1[1]  # B,4,H1*W1,16
        k2, v2 = kv2[0], kv2[1]  # B,4,H2*W2,16

        x0 = self.attention(q0, k0, v0, self.dwc0, B, C // 2, H // self.no_ker_size[0], W // self.no_ker_size[0])
        x1 = self.attention(q1, k1, v1, self.dwc1, B, C // 4, H // self.no_ker_size[1], W // self.no_ker_size[1])
        x2 = self.attention(q2, k2, v2, self.dwc2, B, C // 4, H // self.no_ker_size[2], W // self.no_ker_size[2])

        x = torch.cat([x0, x1, x2], dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CCT_Block(nn.Module):
    r""" Cross-View Cross-Scale Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        no_ker_size (list): Non-overlap kernel sizes in Multi-Scale Aggregation Layer
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 qkv_bias=True,
                 qk_scale=None,
                 no_ker_size=[2, 4, 8]):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(self.dim)

        self.cca = CCAttention(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            attn_drop=attn_drop,
            act_layer=act_layer,
            norm_layer=norm_layer,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            no_ker_size=no_ker_size
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(_init_weights)

    def forward(self, x0, x1):
        H, W = self.input_resolution
        B, L, C = x0.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        shortcut = x0
        x0 = self.norm1(x0)
        x1 = self.norm1(x1)

        # Cross-View Cross-Scale Attention
        x0 = self.cca(x0, x1)

        # FFN
        x0 = shortcut + self.drop_path(x0)
        x0 = x0 + self.drop_path(self.mlp(self.norm2(x0), H, W))

        return x0


class MSA(nn.Module):
    r""" Multi-Scale Aggregation Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Resolution of input feature.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        msa_size (list): Kernel sizes in Multi-Scale Aggregation Layer
        stride (list): Corresponding stride of each conv layer
        split (bool): If features of different scales are returned.  Default: False
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 norm_layer=nn.LayerNorm,
                 msa_size=[3, 5, 9],
                 stride=[1, 1, 1],
                 split=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.reductions = nn.ModuleList()
        self.msa_size = msa_size
        self.stride = stride
        self.norm = norm_layer(self.dim)
        self.split = split
        self.proj_merge = nn.Conv2d(self.dim, self.dim, kernel_size=1) if not split else None

        for i, ps in enumerate(self.msa_size):
            if i == len(self.msa_size) - 1:
                out_dim = 2 * dim // 2 ** (i + 1) if not split else 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 2) if not split else 2 * dim // 2 ** (i + 1)
            padding = (ps - self.stride[i]) // 2
            self.reductions.append(
                nn.Conv2d(
                    dim,
                    out_dim,
                    kernel_size=ps,
                    stride=self.stride[i],
                    padding=padding
                )
            )

        self.apply(_init_weights)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        if self.split:
            # If split features of different scales are needed, return it
            return xs
        x = torch.cat(xs, dim=1)
        x = self.proj_merge(x).flatten(2).permute(0, 2, 1)
        return x


class Stage(nn.Module):
    """ Cross-View Cross-Scale Transformer blocks for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        msa (nn.Module | None, optional): Multi-Scale Aggregation Layer Default: None
        msa_size (list): Kernel sizes in Multi-Scale Aggregation Layer
        no_ker_size (list): Non-overlap kernel sizes in Multi-Scale Aggregation Layer
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth, num_heads,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 qkv_bias=True,
                 qk_scale=None,
                 msa=None,
                 msa_size=[3, 5, 9],
                 no_ker_size=[2, 4, 8]):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                CCT_Block(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    no_ker_size=no_ker_size
                )
            )

        # Multi-Scale Aggregation Layer
        if msa is not None:
            self.msa = msa(dim=dim, input_resolution=input_resolution, norm_layer=norm_layer, msa_size=msa_size)
        else:
            self.msa = None

        self.apply(_init_weights)

    def forward(self, x0, x1):
        for blk in self.blocks:
            # Self-Attention
            x0 = blk(x0, x0)
            x1 = blk(x1, x1)
            # Cross-Attention
            x0 = blk(x0, x1)
            x1 = blk(x1, x0)
        if self.msa is not None:
            x0 = self.msa(x0)
            x1 = self.msa(x1)
        return x0, x1


class CCTransformer(nn.Module):
    r""" Cross-View Cross-Scale Transformer

    Args:
        feat_size (int | tuple(int)): Input feature size. Default 40
        feat_chan (int): Number of input feature channels. Default: 256
        depths (tuple(int)): Depth of each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        msa_sizes (list(list)): Kernel sizes in each stage.
        no_ker_sizes (list(list)): Non-overlap kernel sizes in each stage.
    """

    def __init__(self,
                 feat_size=40,
                 feat_chan=256,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=True,
                 qkv_bias=True,
                 qk_scale=None,
                 msa_sizes=[[3, 5, 9], [3, 5, 9], [3, 5, 9], [3, 5, 9]],
                 no_ker_sizes=[[2, 4, 8], [2, 4, 8], [2, 4, 8], [2, 4, 8]]):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.flat_len = feat_size * feat_size
        self.feat_chan = feat_chan

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.flat_len, self.feat_chan))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            msa_size = msa_sizes[i_layer] if (i_layer < self.num_layers - 1) else None
            no_ker_size = no_ker_sizes[i_layer]
            layer = Stage(
                dim=self.feat_chan,
                input_resolution=(feat_size, feat_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # msa=MSA if (i_layer < self.num_layers - 1) else None,
                msa=None,
                msa_size=msa_size,
                no_ker_size=no_ker_size
            )
            self.layers.append(layer)

        self.apply(_init_weights)

    def forward(self, x0, x1):
        if self.ape:
            x0 = x0 + self.absolute_pos_embed
            x1 = x1 + self.absolute_pos_embed
        x0 = self.pos_drop(x0)
        x1 = self.pos_drop(x1)

        for layer in self.layers:
            x0, x1 = layer(x0, x1)

        return x0, x1
