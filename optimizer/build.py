import re

import torch
from .bert_adam import BertAdam
from util import my_log, overwrite


def init_optimizer(cfg, model, total_step):
    """
    cfg: cfg.optimizer
    """
    global logger, res_logger
    if hasattr(model, 'module'):
        model = model.module

    params = {
        'adapter_decay': [],
        'adapter_no_decay': [],
        'clip_decay': [],
        'clip_no_decay': [],
        'other_decay': [],
        'other_no_decay': [],
    }
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    for n, p in list(model.named_parameters()):
        if 'adapter' in n or 'prompt' in n or 'factor' in n:
            if any([nd in n for nd in no_decay]):
                params['adapter_no_decay'].append(p)
            else:
                params['adapter_decay'].append(p)
        elif 'clip.' in n:
            if any([nd in n for nd in no_decay]):
                params['clip_no_decay'].append(p)
            else:
                params['clip_decay'].append(p)
        else:
            if any([nd in n for nd in no_decay]):
                params['other_no_decay'].append(p)
            else:
                params['other_decay'].append(p)
    
    trainable_type = cfg.get('trainable_type', ['.*'])
    cfg.trainable_type = [k for k in params if any([re.match(f'{t}$', k) for t in trainable_type])]
    
    for n, p in params.items():
        if n not in cfg.trainable_type:
            for val in p:
                val.requires_grad = False

    trainable_params = sum([p.numel() for n, p in model.named_parameters() if p.requires_grad])
    clip_params = sum([p.numel() for p in params['clip_decay'] + params['clip_no_decay']])
    clip_nd_params = sum([p.numel() for p in params['clip_no_decay']])
    clip_d_params = sum([p.numel() for p in params['clip_decay']])
    other_params = sum([p.numel() for p in params['other_decay'] + params['other_no_decay']])
    adapter_params = sum([p.numel() for p in params['adapter_decay'] + params['adapter_no_decay']])
    total_params = clip_params + other_params + adapter_params
    trainable_ratio = (trainable_params / clip_params) * 100

    param_log = '\nclip params: {}\n\tclip decay params: {}\n\t' + \
                'clip no decay params: {}, \nadapter params: {}\n' + \
                'other params: {}\ntotal params: {}\n' + \
                'trainable params: {}, ratio: {:.3f}%'
    my_log(param_log.format(
        clip_params, clip_d_params, clip_nd_params,
        adapter_params, other_params, total_params,
        trainable_params, trainable_ratio
    ))

    default = dict(weight_decay=0, lr=1e-7,)
    default = overwrite(default, cfg.get('default', {}))

    optimizer_grouped_parameters = []
    for t in cfg.trainable_type:
        group_cfg = overwrite(default, cfg.get(t, {}))
        optimizer_grouped_parameters.append({'params': params[t], **group_cfg})

    optimizer = BertAdam(optimizer_grouped_parameters, lr=default['lr'], warmup=cfg.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6, max_grad_norm=1.0,
                         t_total=total_step, weight_decay=default['weight_decay'])

    
    return optimizer
