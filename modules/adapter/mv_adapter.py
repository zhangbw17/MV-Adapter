import math
from typing import OrderedDict

import torch
from torch import nn, Tensor

from registry import ADAPTERS
from modules.module_ta import TaModuleV1, TaModuleV2
from mmengine.runner.amp import autocast


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform quick gelu."""
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@ADAPTERS.register_module()
class VideoAdapter(nn.Module):
    def __init__(
        self, embed_dim=512, cls_mid=64, n_head=2,
        attn_type=None, scale=0.1, pos=None, idx=0, seq_len=12, 
        pca='pca', calibrate_func='v1', ratio=[0.5, 0.5], **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_mid = cls_mid
        self.n_head = n_head
        self.temporal = kwargs.get('temporal', [])
        assert all([t in ['p', 'c'] for t in self.temporal])
        self.idx = idx
        self.no_cc = kwargs.get('no_cc', False)

        self.ln_in_flag = kwargs.get('lnin', False)
        if self.ln_in_flag:
            self.ln_in = LayerNorm(embed_dim)

        cal_func = {
            'v1': TaModuleV1,
            'v2': TaModuleV2,
        }
        if 'c' in self.temporal:
            self.conv_rf_c = cal_func[calibrate_func](
                c_in=cls_mid,            # number of input filters
                c_out=cls_mid,
                cc_dim=0 if self.no_cc else cls_mid,
                mid_dim=16,            # reduction ratio for MLP
                kernels=3,      # list of temporal kernel sizes
                concat=not self.no_cc
            )
        if 'p' in self.temporal:
            self.conv_rf_p = cal_func[calibrate_func](
                c_in=cls_mid,            # number of input filters
                c_out=cls_mid,
                cc_dim=0 if self.no_cc else cls_mid,
                mid_dim=16,            # reduction ratio for MLP
                kernels=3,      # list of temporal kernel sizes
                concat=not self.no_cc
            )

        self.down = nn.Linear(embed_dim, cls_mid)
        self.up = nn.Linear(cls_mid, embed_dim)

        self.act = QuickGELU()
        self.scale = scale
        self.ln_pre = LayerNorm(cls_mid)
        self.block = AdapterTransBlock(cls_mid, n_head, attn_type)

        self.positional_embedding = nn.Parameter(torch.randn(seq_len * 2 + 1, cls_mid))
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.cc = nn.Parameter(cls_mid ** -.5 * torch.randn(cls_mid))
        self.patch_pooling = pca

        if ratio[1] is None:
            self.ratio_p = nn.Parameter(torch.tensor(0.))
        else:
            self.ratio_p = ratio[1]
                    
        self.modal_dynamic = False
        if kwargs.get('modal_dynamic', {}):
            md_cfg = kwargs['modal_dynamic']
            if idx in md_cfg['adapter_idx']:
                self.modal_dynamic = True
                md_idx = md_cfg['adapter_idx'].index(idx)
                self.shared_factor = md_cfg['factor'](md_idx)
                self.md_pos = md_cfg['position']
                if 'up in' in md_cfg['position'] or 'down out' in md_cfg['position']:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], cls_mid)
                else:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], embed_dim)
        self.init()

    def forward(self, x: Tensor, casual_mask, **kwargs):
        """
        input shape v16 197 * (bs * frames) * 512
        cls 1 * (bs * frames) * 512
        video input: cls bs * frames
        """
        # ----- utils ----- #
        B, F = casual_mask.shape
        patch_num = int((x.shape[0] - 1) ** .5)
        # ----- utils ----- #

        # ----- downsample the cls and patch ----- #
        if self.ln_in_flag:
            x = self.ln_in(x)

        down_weight = self.down.weight
        if self.modal_dynamic and self.md_pos.startswith('down'):
            factor = self.md_proj(self.shared_factor)                
            if 'down in' in self.md_pos:
                down_weight = torch.einsum('i,oi->oi', factor, down_weight)
            elif 'down out' in self.md_pos:
                down_weight = torch.einsum('o,oi->oi', factor, down_weight)

        if self.modal_dynamic and self.md_pos in ['down in cls', 'down out cls']:
            cls_emb = x[0].reshape(B, -1, self.embed_dim)
            patch_emb = x[1:].reshape(patch_num ** 2, B, -1, self.embed_dim)
            cls_down = self.act(cls_emb @ down_weight.T + self.down.bias)
            patch_down = self.act(self.down(patch_emb))
        else:
            down = x @ down_weight.T + self.down.bias
            down = self.act(down)
            cls_down = down[0].reshape(B, -1, self.cls_mid)
            patch_down = down[1:].reshape(patch_num ** 2, B, -1, self.cls_mid) 
        # ----- downsample the cls and patch ----- #
        
        # ----- use pca or mean pooling to down sample patch ----- #
        if self.patch_pooling == 'pca':
            patch_emb_bnel = patch_down.permute(1, 2 ,3, 0)  # L * B * N * E -> B * N * E * L
            with autocast(dtype=torch.float32):
                u, s, v = torch.pca_lowrank(patch_emb_bnel, 1)
            patch_pca = torch.matmul(patch_emb_bnel, v).reshape(B, F, -1)
        elif self.patch_pooling == 'tconv':
            patch_2d = patch_down.reshape(patch_num, patch_num, B, F, -1)
            patch_bcfpp = patch_2d.permute(2, 4, 3, 0, 1)   # B, E, frame, pn, pn
            offset = (F + 1) % 2
            patch_pca = self.tconv(patch_bcfpp)[:, :, offset:]     # B, E, fn, pn, pn
            patch_pca = patch_pca.reshape(B, -1, F, patch_num ** 2)
            patch_pca = patch_pca.mean(dim=-1).permute(0, 2, 1)     # B, F, E
        else:
            patch_pca = patch_down.mean(dim=0)
        # ----- use pca or mean pooling to down sample patch ----- #

        # ----- temporal modeling ----- #
        dt, device = cls_down.dtype, cls_down.device
        cls_mid = cls_down.shape[-1]
        if self.no_cc:
            temporal_seq = torch.cat([cls_down, patch_pca], dim=1)
        else:
            cc = self.cc.to(dt) + torch.zeros(B, 1, cls_mid, dtype=dt, device=device)
            temporal_seq = torch.cat([cc, cls_down, patch_pca], dim=1)
        pos_emd = self.positional_embedding[:temporal_seq.size(1), :].to(x.dtype)
        temporal_seq = temporal_seq + pos_emd

        temporal_seq = self.ln_pre(temporal_seq)
        temporal_seq = temporal_seq.permute(1, 0, 2)
        # ----- temporal modeling ----- #

        # ----- get attention mask for video ----- #
        # TODO
        v_mask = casual_mask
        mask = torch.cat([torch.ones(B, int(not self.no_cc)).to(device), casual_mask.repeat(1, 2)], dim=1)
        e_mask = (1.0 - mask.unsqueeze(1)) * -1000000.0
        e_mask = e_mask.expand(-1, mask.size(1), -1)
        attn_mask_ = e_mask.repeat_interleave(self.n_head, dim=0)
        # ----- get attention mask for video ----- $

        # ----- temporal modeling for cls ----- #
        temporal_seq = self.block(temporal_seq, attn_mask_)
        temporal_seq = temporal_seq.permute(1, 0, 2)    # bs, frames + 1, e_a
        temporal_seq = self.act(temporal_seq)
        
        if self.no_cc:
            cc_temp = 0
            cls_temp = temporal_seq[:, :F]
            patch_temp = temporal_seq[:, F:]
        else:
            cc_temp = temporal_seq[:, 0]
            cls_temp = temporal_seq[:, 1:F+1]
            patch_temp = temporal_seq[:, F+1:]
        # ----- temporal modeling for cls ----- #

        # ----- patch for patch calibrate ----- #
        if 'p' in self.temporal:
            calibrate_input = cls_temp + patch_temp
            
            alpha_p = self.conv_rf_p(cc_temp, calibrate_input, v_mask)
            delta_patch = torch.einsum('bfi,oi,pbfi->pbfo', alpha_p, self.up.weight, patch_down)
            delta_patch = delta_patch.reshape(patch_num ** 2, -1, self.embed_dim)
        else:
            delta_patch = torch.einsum('pbfi,oi->pbfo', patch_down, self.up.weight).reshape(patch_num ** 2, -1, self.embed_dim)
        patch_up = delta_patch + self.up.bias
        # ----- patch for patch calibrate ----- #

        # ----- cls up sample ----- #
        up_weight = self.up.weight
        if self.modal_dynamic and self.md_pos.startswith('up'):
            factor = self.md_proj(self.shared_factor)
            if self.md_pos == 'up in':
                up_weight = torch.einsum('i,oi->oi', factor, up_weight)
            elif self.md_pos == 'up out':
                up_weight = torch.einsum('o,oi->oi', factor, up_weight)
        if 'c' in self.temporal:
            calibrate_input = cls_temp + patch_temp
            alpha_c = self.conv_rf_c(cc_temp, calibrate_input, v_mask)
            cls_up = torch.einsum('bfi,oi,bfi->bfo', alpha_c, up_weight, cls_temp)
        else:
            cls_up = cls_temp @ up_weight.T
        cls_up = (cls_up + self.up.bias).reshape(1, -1, self.embed_dim)
        # ----- cls up sample ----- #

        delta_x = self.scale * torch.cat([cls_up, patch_up])
        return delta_x
    
    def init(self):
        proj_std = ((2 *self.embed_dim) ** -0.5)
        attn_std = self.embed_dim ** -0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        nn.init.normal_(self.block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(self.block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.block.mlp.c_proj.weight, std=proj_std)


        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)



@ADAPTERS.register_module()
class TextTransfAdapter(nn.Module):
    def __init__(
        self, embed_dim=512, mid_dim=64, n_head=2, idx=-1,
        attn_type=None, scale=0.1, pos=None, seq_len=12,
        pca=False, **kwargs
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.n_head = n_head
        self.idx = idx
        self.up_share, self.down_share = False, False

        self.pca = pca
        if pca:
            def pca_down(x):
                seq_len, bs, _ =  x.shape
                group_dim = int(embed_dim // mid_dim)
                assert group_dim * mid_dim == embed_dim
                x = x.permute(1, 0, 2)
                x = x.reshape(bs, seq_len, mid_dim, group_dim)   # bs * seq * dim
                pca_dim = min(*x.shape[-2:], mid_dim)
                u, s, v = torch.pca_lowrank(x.float(), pca_dim)
                x_down = torch.matmul(x, v[:, :, :, :1].half()).reshape(bs, seq_len, mid_dim)
                x_down = x_down.permute(1, 0, 2)
                return x_down
            self.down = pca_down
        else:
            self.down = nn.Linear(embed_dim, mid_dim)
        self.up = nn.Linear(mid_dim, embed_dim)
        self.act = QuickGELU()
        self.scale = scale
        self.ln_pre = LayerNorm(mid_dim)
        self.block = AdapterTransBlock(mid_dim, n_head, attn_type=attn_type)
        self.positional_embedding = nn.Parameter(torch.randn(seq_len, mid_dim))
        if pos is not None:
            self.positional_embedding = nn.Parameter(pos.clone()[:,:mid_dim])
        else:
            nn.init.normal_(self.positional_embedding, std=0.01)

        self.ln_in_flag = kwargs.get('lnin', False)
        if self.ln_in_flag:
            self.ln_in = LayerNorm(embed_dim)
    
        self.modal_dynamic = False
        if kwargs.get('modal_dynamic', {}):
            md_cfg = kwargs['modal_dynamic']
            if idx in md_cfg['adapter_idx']:
                self.modal_dynamic = True
                md_idx = md_cfg['adapter_idx'].index(idx)
                self.shared_factor = md_cfg['factor'](md_idx)
                self.md_pos = md_cfg['position']
                if md_cfg['position'] in ['up in', 'down out']:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], mid_dim)
                else:
                    self.md_proj = nn.Linear(self.shared_factor.shape[0], embed_dim)
        self.init()    
        
    def forward(self, x: Tensor, *args, **kwargs):
        if self.ln_in_flag:
            x = self.ln_in(x)

        down_weight = self.down.weight
        if self.modal_dynamic and self.md_pos.startswith('down'):
            factor = self.md_proj(self.shared_factor)
            if self.md_pos == 'down in':
                down_weight = torch.einsum('i,oi->oi', factor, down_weight)
            elif self.md_pos == 'down out':
                down_weight = torch.einsum('o,oi->oi', factor, down_weight)
        down = x @ down_weight.T + self.down.bias

        pos_emd = self.positional_embedding[:, None].to(x.dtype)
        down = down + pos_emd
        
        down = self.ln_pre(down)

        down = self.block(down)
        down = self.act(down)
        
        up_weight = self.up.weight
        if self.modal_dynamic and self.md_pos.startswith('up'):
            factor = self.md_proj(self.shared_factor)
            if self.md_pos == 'up in':
                up_weight = torch.einsum('i,oi->oi', factor, up_weight)
            elif self.md_pos == 'up out':
                up_weight = torch.einsum('o,oi->oi', factor, up_weight)
        up = down @ up_weight.T + self.up.bias
        delta_x = self.scale * up
        return delta_x

    def init(self):
        proj_std = ((2 *self.embed_dim) ** -0.5)
        attn_std = self.embed_dim ** -0.5
        fc_std = (2 * self.embed_dim) ** -0.5
        nn.init.normal_(self.block.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.block.attn.out_proj.weight, std=proj_std)
        nn.init.normal_(self.block.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.block.mlp.c_proj.weight, std=proj_std)

        if not self.pca:
            nn.init.normal_(self.down.weight, std=fc_std)
            nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)


class AdapterTransBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_type=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_type = attn_type

    def make_attn_mask(self, x):
        attn_mask = None
        if self.attn_type == 'uni':
            attn_mask = torch.zeros(x.size(0), x.size(0)).to(x.device)
            attn_mask.fill_(float("-inf"))
            attn_mask.triu_(1)  # zero out the lower diagonal
        return attn_mask

    def forward(self, x, attn_mask=None):
        if attn_mask is None:
            attn_mask = self.make_attn_mask(x)
        
        ln_x = self.ln_1(x)
        x = x + self.attn(ln_x, ln_x, ln_x, need_weights=False, attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x
