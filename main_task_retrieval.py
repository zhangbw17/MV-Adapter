from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import random
import os
from os import path as osp
import argparse
import time

import torch
import numpy as np
import torch.distributed as dist
from mmengine import Config, DictAction

from optimizer import init_optimizer
from registry import RUNNERS
from runner import RetrievalRunner
from modules import init_model
from util import Loggers, get_logger, my_log
from dataloaders.data_dataloaders import build_loader


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--work-root', default=None, 
                        help='the dir of pretrained models and log')
    parser.add_argument(
        '--checkpoint', default=None,
        help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--fp32', action='store_true', help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    cfg.git_id = os.popen('git rev-parse HEAD').read().strip()
    cfg.git_msg = os.popen('git log --pretty=format:"%s" {} -1'.format(cfg.git_id)).read().strip()
    cfg.trial_id = os.popen('echo $ARNOLD_TRIAL_ID').read().strip()

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # resume training
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint

    # set random seeds
    if args.seed is not None:
        cfg.seed = args.seed

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.local_rank = args.local_rank
    cfg.work_dir = osp.join(cfg.work_dir, str(cfg.seed))
    cfg.model.fp32 = args.fp32
    if args.fp32:
        cfg.train_dataloader.batch_size //= 2
        cfg.val_dataloader.batch_size //= 2
    
    if args.work_root is not None:
        current_root = './'
        new_root = args.work_root
        cfg.work_dir = cfg.work_dir.replace(current_root, new_root)
        cfg.train_dataset.data_root = cfg.train_dataset.data_root.replace(current_root, new_root)
        cfg.test_dataset.data_root = cfg.test_dataset.data_root.replace(current_root, new_root)
        cfg.model.clip_cache_dir = cfg.model.clip_cache_dir.replace(current_root, new_root)
    
    if dist.get_rank() == 0:
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(args.config) as cfg_file:
            open(osp.join(
                cfg.work_dir, f'{osp.basename(args.config)}'
            ), 'a+').write(cfg_file.read())
    return cfg


def set_seed_logger(cfg):
    # predefining random initial seeds
    random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = dist.get_world_size()
    torch.cuda.set_device(cfg.local_rank)
    cfg.world_size = world_size
    rank = dist.get_rank()
    cfg.rank = rank

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = {
        'default': osp.join(cfg.work_dir, timestamp),
        'loss': osp.join(cfg.work_dir, 'loss'),
        'result': osp.join(cfg.work_dir, 'result'),
    }
    
    for n, l in log_file.items():
        if dist.get_rank() == 0:
            Loggers.loggers[n] = get_logger(l, n, not n=='default')
    my_log(
        'CONFIG:\n' + '\n'.join(["{}: {}".format(key, cfg._cfg_dict[key]) 
        for key in sorted(cfg._cfg_dict.keys())])
    )
    return cfg


def main():
    args = get_args()
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg = set_seed_logger(cfg)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", cfg.local_rank)

    model = init_model(cfg, device)

    ## ####################################
    # dataloader loading
    ## ####################################
    train_dataloader, val_dataloader = build_loader(cfg)
    optimizer = init_optimizer(cfg.optimizer, model, cfg.total_step)

    ## ####################################
    # train and eval
    ## ####################################
    runner_cls = RUNNERS.get(cfg.get('runner', 'RetrievalRunner')) 
    runner = runner_cls(cfg, model, train_dataloader, val_dataloader, optimizer)
    runner.run()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    main()
