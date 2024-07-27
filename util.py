from typing import List
import threading
import logging
import datetime
from copy import deepcopy

import torch
import torch.nn as nn
from torch._utils import ExceptionWrapper
import torch.distributed as dist


class Loggers:
    loggers = {}


def my_log(msg, level=logging.INFO, logger='default'):
    if dist.get_rank() == 0:
        Loggers.loggers['default'].log(level, msg)
        if not logger == 'default':
            Loggers.loggers[logger].log(level, msg)


def overwrite(default_dict, new_dict):
    out = deepcopy(default_dict)
    for k, v in new_dict.items():
        out[k] = v
    return out


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

def parallel_apply(fct, model, inputs, device_ids):
    modules = nn.parallel.replicate(model, device_ids)
    assert len(modules) == len(inputs)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input):
        torch.set_grad_enabled(grad_enabled)
        device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = fct(module, *input)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input))
                   for i, (module, input) in enumerate(zip(modules, inputs))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs

def get_logger(filename=None, logger_name='logger', no_terminal=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger.root.handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(stream_handler)
    if filename is not None:
        handler = logging.FileHandler(filename + '.log')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        if no_terminal:
            logger.handlers = []
        logger.addHandler(handler)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.avg)
        # for stats
        return 'cur: %.6f, last %d step avg: (%.4f)' % (self.val, self.count, self.avg)


def my_all_gather(obj):
    output = [
        torch.Tensor() for _ in range(dist.get_world_size())
    ]
    gather = dist.all_gather_object
    gather(output, obj)
    return output


def cpu_cat(tensors: List[torch.Tensor], *args, **kwargs):
    return torch.cat([t.detach().cpu() for t in tensors], *args, **kwargs)


def merge_func(ele):
    if isinstance(ele[0], list or tuple):
        return [oo for o in ele for oo in o]
    elif isinstance(ele[0], dict):
        return {k: v for o in ele for k, v in o.items()}
    elif isinstance(ele[0], torch.Tensor):
        return torch.cat(ele)
    return ele

