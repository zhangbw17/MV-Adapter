import math
from typing import Tuple, Union, List, Optional
import warnings
from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mmengine.model import BaseModel
from registry import ADAPTERS


class Zero(nn.Module):
    def forward(self, *args, **kwargs) -> Tensor:
        return 0


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, d_model, n_head, dropout=0., adapter={}, idx=-1):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = n_head

        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        forward_dict = {
            'default': self.base_forward,
        }

        adapter = deepcopy(adapter)
        forward_key = adapter.pop('forward', 'default')
        self.forward_func = self.base_forward
        linear_module = nn.Linear

        if not adapter or idx not in adapter['adapter_idx']:
            self.forward_func = self.base_forward
            adapter = {}
        else:
            if adapter:
                model_type = adapter.pop('type')
                model_cls = ADAPTERS.get(model_type)
                adapter['idx'] = idx
                adapter['pos'] = adapter.get('pos', None)
                self.adapter = model_cls(**adapter)

        self.q_proj = linear_module(self.embed_dim, self.embed_dim, **adapter.get('q', {}))
        self.k_proj = linear_module(self.embed_dim, self.embed_dim, **adapter.get('k', {}))
        self.v_proj = linear_module(self.embed_dim, self.embed_dim, **adapter.get('v', {}))
        self.out_proj = linear_module(self.embed_dim, self.embed_dim, **adapter.get('o', {}))
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        return self.forward_func(*args, **kwargs)

    def base_forward(self, x, attn_mask, *args, **kwargs):
        ln_x = self.ln(x)
        out = x + self.mha(ln_x, ln_x, ln_x, need_weights=False, attn_mask=attn_mask)[0]
        return out

    def mha(self, q, k, v, attn_mask, need_weights=True, use_proj=True):
        tgt_len, bsz, embed_dim = q.shape
        src_len, _, _ = k.shape
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads

        # get query proj
        if use_proj:
            # do not use proj if some processes have done and get the projected results
            q = self.q_proj(q) * self.scale
            k = self.k_proj(k)
            v = self.v_proj(v)
       
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        attn_mask = self.attn_mask(bsz, tgt_len, src_len, attn_mask)

        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)

        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        output = torch.bmm(attn, v)

        attn_output = output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

    def attn_mask(self, bsz, tgt_len, src_len, attn_mask=None):
        # apply the causal_attention_mask first
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
        return attn_mask


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CLIPMlp(BaseModel):
    def __init__(self, d_model, adapter={}, idx=-1) -> None:
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)
        self.ln = nn.LayerNorm(d_model)

        forward_dict = {
            'default': self.forward_default,
            'seq': self.forward_seq,
            'par': self.forward_parallel,
            'ln_par': self.forward_ln_parallel,
        }
        adapter = deepcopy(adapter)
        forward_key = adapter.pop('forward', 'default')
        self.forward_func = forward_dict[forward_key]

        if idx not in adapter.get('adapter_idx', list(range(12))):
            self.forward_func = self.forward_default
            adapter = None
    
        if adapter:
            model_type = adapter.pop('type')
            model_cls = ADAPTERS.get(model_type)
            adapter['idx'] = idx
            adapter['pos'] = adapter.get('pos', None)
            self.adapter = model_cls(**adapter)

    def forward(self, *args, **kwargs):
        return self.forward_func(*args, **kwargs)

    def forward_seq(self, x, *args, **kwargs):
        resdiual = x
        x = self.ln(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        ax = self.adapter(x, *args, **kwargs)
        out = x + resdiual + ax
        return out
    
    def forward_parallel(self, x, *args, **kwargs):
        resdiual = x
        ax = self.adapter(x, *args, **kwargs)
        x = self.ln(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        out = x + resdiual + ax
        return out
    
    def forward_ln_parallel(self, x, *args, **kwargs):
        resdiual = x
        x = self.ln(x)
        ax = self.adapter(x, *args, **kwargs)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        out = x + resdiual + ax
        return out

    def forward_default(self, x, *args, **kwargs):
        resdiual = x
        x = self.ln(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        out = x + resdiual
        return out
