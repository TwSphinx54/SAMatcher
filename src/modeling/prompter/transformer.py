import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from timm.layers import trunc_normal_
from src.modeling.prompter.CCT import CCAttention, Mlp, _init_weights


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum('nshd,nshv->nhdv', K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum('nlhd,nhdv,nlh->nlhv', Q, KV, Z) * v_length

        return queried_values.contiguous()


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True,
                 kdim=None,
                 vdim=None):
        super(MultiHeadAttention, self).__init__()

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.nhead = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert (self.head_dim * num_heads == self.embed_dim), 'embed_dim must be divisible by num_heads'

        # multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.attention = LinearAttention()
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)

        self.apply(_init_weights)

    def forward(self, q, k, v):
        bs = q.size(0)
        # multi-head attention
        # [N, L, (H, D)]
        query = self.q_proj(q).view(bs, -1, self.nhead, self.head_dim)
        # [N, S, (H, D)]
        key = self.k_proj(k).view(bs, -1, self.nhead, self.head_dim)
        value = self.v_proj(v).view(bs, -1, self.nhead, self.head_dim)
        # [N, L, (H, D)]
        message = self.attention(query, key, value)
        # [N, L, C]
        message = self.merge(message.view(bs, -1, self.nhead * self.head_dim))

        return message


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 feat_size,
                 no_ker_size,
                 mlp_ratio=4.,
                 dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.feat_size = feat_size
        self.mlp_ratio = mlp_ratio
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, (feat_size ** 2), self.dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.self_attn = MultiHeadAttention(self.dim, nhead)

        self.cca = CCAttention(
            dim=self.dim,
            input_resolution=(feat_size, feat_size),
            num_heads=self.nhead,
            qkv_bias=True,
            qk_scale=False,
            no_ker_size=no_ker_size
        )
        # self.multihead_attn = MultiHeadAttention(d_model, nhead)

        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim)

        # norm and dropout
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.norm3 = nn.LayerNorm(self.dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.apply(_init_weights)

    def forward(self,
                num_points,
                tgt,
                memory,
                tgt_pos=None):
        """
        Args:
            :param tgt: (torch.Tensor): [N, L, C]
            :param memory: (torch.Tensor): [N, S, C]
            :param tgt_pos:
        """
        # Query Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = with_pos_embed(tgt2, tgt_pos)
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)

        # Object Query
        tgt2 = self.norm2(tgt)
        memory = memory + self.absolute_pos_embed
        tgt2 = self.cca(tgt2, memory)
        # tgt2 = self.multihead_attn(
        #     q=with_pos_embed(tgt2, tgt_pos),
        #     k=with_pos_embed(memory, None),
        #     v=memory
        # )
        tgt = tgt + self.dropout2(tgt2)
        tgt = tgt + self.mlp(self.norm3(tgt), num_points, 1)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, feat_size, no_ker_size, num_layers, norm=None):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = DecoderLayer(
                d_model,
                nhead,
                feat_size,
                no_ker_size,
                dropout=0.1
            )
            self.layers.append(layer)
        self.norm = norm

        self.apply(_init_weights)

    def forward(self,
                num_points,
                tgt,
                memory,
                tgt_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                num_points,
                output,
                memory,
                tgt_pos=tgt_pos
            )
        if self.norm is not None:
            output = self.norm(output)
        return output
