import torch
import torch.nn as nn



class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform quick gelu."""
        return x * torch.sigmoid(1.702 * x)


class TaModuleV1(nn.Module):
    def __init__(self, c_in, cc_dim, c_out, mid_dim, concat=True, **kwargs):
        super(TaModuleV1, self).__init__()
        self.c_in = c_in
        self.concat = concat
        self.avgpool = nn.AdaptiveAvgPool3d((None,1,1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.cc_dim = cc_dim
        if cc_dim:
            self.g = nn.Linear(cc_dim, cc_dim)
        else:
            self.g = nn.Identity()
        self.a = nn.Linear(c_in + cc_dim if self.concat else c_in, mid_dim)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Linear(mid_dim, c_out, bias=False)
        self.b.weight.data.zero_()
        
    def forward(self, cc_emb, cls_emb, v_mask):
        if self.concat:
            calibrate_emb = torch.cat([cls_emb, self.g(cc_emb)[:, None].expand(-1, cls_emb.shape[1], -1)], dim=-1)
        else:
            if self.cc_dim: calibrate_emb = cls_emb + self.g(cc_emb)[:, None]
            else: calibrate_emb = cls_emb
        calibrate_emb = torch.einsum('bsc,bs->bsc', calibrate_emb, v_mask)
        x = self.a(calibrate_emb)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x


class TaModuleV2(nn.Module):
    def __init__(self, c_in, cc_dim, c_out, mid_dim, kernels, concat=True):
        super(TaModuleV1, self).__init__()
        self.c_in = c_in
        self.concat = concat
        self.cc_dim = cc_dim
        if cc_dim:
            self.g = nn.Linear(cc_dim, cc_dim)
        else:
            self.g = nn.Identity()
        self.a = nn.Conv1d(
            c_in + cc_dim if self.concat else c_in, 
            mid_dim,
            kernel_size=kernels,
            padding=1,
        )
        self.norm = nn.LayerNorm(mid_dim)
        self.act = QuickGELU()
        self.b = nn.Conv1d(
            in_channels=mid_dim,
            out_channels=c_out,
            kernel_size=kernels,
            padding=1,
            bias=False
        )
        
    def forward(self, cc_emb, cls_emb, v_mask):
        if self.concat:
            calibrate_emb = torch.cat([cls_emb, self.g(cc_emb)[:, None].expand(-1, cls_emb.shape[1], -1)], dim=-1)
        else:
            calibrate_emb = cls_emb + self.g(cc_emb)[:, None]
            if self.cc_dim: calibrate_emb = cls_emb + self.g(cc_emb)[:, None]
            else: calibrate_emb = cls_emb
        calibrate_emb = torch.einsum('bsc,bs->bsc', calibrate_emb, v_mask).permute(0, 2, 1)
        x = self.a(calibrate_emb)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.act(x)
        x = self.b(x).softmax(dim=-1) * x.shape[-1]
        return x.permute(0, 2, 1)

