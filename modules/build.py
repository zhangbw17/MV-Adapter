
import torch
from torch import nn

from .modeling import BaselineModel
from registry import MODELS
    

def init_model(cfg, device):

    model_cfg = cfg.model
    model_type = model_cfg.pop('type')
    model_cls = MODELS.get(model_type)
    model = model_cls(**model_cfg)
    if cfg.checkpoint:
        model_state_dict = torch.load(cfg.checkpoint, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank],
                                                      output_device=cfg.local_rank, find_unused_parameters=True)

    return model

