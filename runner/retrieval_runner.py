import math
from typing import Dict, Union
from torch.functional import Tensor
from tqdm import tqdm
import os.path as osp
from time import time

import torch
from torch import distributed as dist
import numpy as np

from .base_runner import Runner
from modules.until_module import AllGather
from util import my_log
from util import AverageMeter, my_log, cpu_cat, merge_func, my_all_gather
from registry import RUNNERS

allgather = AllGather.apply


@RUNNERS.register_module()
class RetrievalRunner(Runner):
    def __init__(self, cfg, model, train_dataloader, val_dataloader, optimizer) -> None:
        super().__init__(cfg, model, train_dataloader, val_dataloader, optimizer)
        self.results = []
        self.loss_meter = AverageMeter()
        self.time_meter = AverageMeter()
        self.f_meter = AverageMeter()
        self.b_meter = AverageMeter()
    
    def train(self): 
        model = self.model
        model.train()
        torch.cuda.empty_cache()
        log_step = self.cfg.log_step

        start_time = time()
        train_dl = tqdm(self.train_dataloader) if dist.get_rank() == 0 else self.train_dataloader
        for step, batch in enumerate(train_dl):
            t1 = time()
            loss = model(batch)
            self.f_meter.update(time() - t1)

            self.loss_meter.update(loss.item())
            # {n: dict(type=p.dtype, p=p, grad=p.grad, rp=p.requires_grad) for n, p in model.named_parameters()}
            t1 = time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.b_meter.update(time() - t1)

            # https://github.com/openai/CLIP/issues/46
            if self.cfg.tau == 'sqrt':
                max_tau = 100 - 80 * np.sqrt(step / self.max_steps)
            else: 
                max_tau = 100
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(max_tau))
            else: torch.clamp_(model.clip.logit_scale.data, max=np.log(max_tau))

            self.global_step += 1
            if self.global_step % log_step == 0:
                estimated_time = sec2timestamp(self.time_meter.avg * (self.max_steps - self.global_step))
                # tb_logger.log('loss', loss_meter.avg, global_step)
                train_info = f"\tEpoch: {self.epoch + 1}/{self.max_epochs}, Step: {step + 1}/{len(self.train_dataloader)},"\
                             f"Lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}" \
                             f"Loss: {str(self.loss_meter)}, \n\t\t\t\tTime/step: {self.time_meter.avg}, Estimated: {estimated_time}, " \
                             f"max memeory: {torch.cuda.max_memory_allocated() // 1024**2}M"
                my_log(train_info, logger='loss')
                time_info = f'Forward: {self.f_meter.avg}, Backward: {self.b_meter.avg}'
                my_log(time_info, logger='loss')
                self.loss_meter.reset()
                self.time_meter.reset()
                self.f_meter.reset()
                self.b_meter.reset()
            self.time_meter.update(time() - start_time)
            start_time = time()

    def run(self):
        cfg = self.cfg.train_cfg
        if cfg.val_begin == 0:
            self.evaluate()

        model = self.model
        start = time()
        for e in range(cfg.max_epochs):
            self.epoch = e
            self.train_dataloader.sampler.set_epoch(e)
            model.epoch = e
            # with torch.autograd.set_detect_anomaly(True):
            self.train()
            
            res = self.evaluate()
            if dist.get_rank() > 0:
                continue
            res['model'] = self.save_model(res['t2v_r1'], res['v2t_r1'])
            self.results.append(res)

            my_log(
                f'Epoch {e + 1}/{self.max_epochs} Finished in {sec2timestamp(time() - start)}',
                logger='result'
            )
            start = time()
        if dist.get_rank() == 0: 
            self.log_best()

    @torch.no_grad()
    def evaluate(self):
        model = self.model
        model.eval()
        # v_emb_list, t_emb_list, v_mask_list, t_mask_list = [], [], [], []
        res = []
        fp_list, captions = [], []
        t2v_label = []

        rank, world_size = dist.get_rank(), dist.get_world_size()
        item_num = len(self.val_dataloader.dataset)
        full_batch = item_num / world_size
        padded_items = math.ceil(full_batch) * world_size - item_num
        print(time())
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            if i == len(self.val_dataloader) - 1 and world_size - rank <= padded_items:
                batch = {k: v[:-1] for k, v in batch.items()}
            fp_list += [fp[0].split('/')[-1].split('.')[0] for fp in batch['image_paths']]
            v_mask = batch['v_mask']
            if not batch['caption_batch']:
                continue
            captions += batch['caption_batch']
            joined_captions = "\n".join(batch['caption_batch'])
            # print(f'rank: {rank}, idx: {i}, captions: \n{joined_captions}\n-----------\n')
            if isinstance(batch['caption_batch'][0], list): 
                for cap in batch['caption_batch']:
                    cur_label = t2v_label[-1] + 1 if t2v_label else 0
                    t2v_label += [cur_label for _ in cap]
                batch['caption_batch'] = [cap for caps in batch['caption_batch'] for cap in caps]
            res.append(model(batch))
        # calculate similarity
        print(time())
        rank_res = [merge_func(r).cpu() for r in list(zip(*res))]
        t2v_label = [l for rank_label in my_all_gather(t2v_label) for l in rank_label]

        t_emb, v_emb, t_mask, v_mask = [torch.cat(my_all_gather(r)) for r in rank_res]
        # for msvd

        if not dist.get_rank() == 0:
            return None
        
        if t2v_label: 
            offset = 0
            new_t2v_label = []
            for i, l in enumerate(t2v_label):
                if i > 0 and l == 0 and t2v_label[i - 1] > 0:
                    offset = new_t2v_label[i - 1] + 1
                new_t2v_label.append(l + offset)
            t2v_label = new_t2v_label

            last_label, s = 0, 0
            v2t_label = torch.zeros(v_mask.shape[0], t_mask.shape[0])
            for i, l in enumerate(t2v_label):
                if not l == last_label:
                    v2t_label[last_label, s:i] = 1
                    last_label = l
                    s = i
            v2t_label[last_label, s:] = 1
            t2v_label = torch.tensor(t2v_label)
        else:
            v2t_label = None

        # if cuda out of memory, make chunk_size smaller
        t2v_list, v2t_list, chunk_idx = [], [], 0
        if hasattr(model, 'module'): model = model.module
        t2v_logits, v2t_logits = [
            l.detach().cpu() for l in 
            model.get_similarity_logits(t_emb, v_emb, t_mask, v_mask)
        ]
        
        t2v_res = get_rank(t2v_logits, label=t2v_label)
        v2t_res = get_rank(v2t_logits, prefix='v2t', label=v2t_label)

        res = dict(t2v_res.items() | v2t_res.items())
        t2v_info = f"Epoch: {self.epoch} >>>  Text-to-Video:\n" \
                   f"R@1: {res['t2v_r1']:.3f} - R@5: {res['t2v_r5']:.3f} - R@10: {res['t2v_r10']:.3f}\n" \
                   f"Sum: {res['t2v_sum']:.3f} - Median R: {res['t2v_median']:.1f} - Mean R: {res['t2v_mean']:.1f}"
        v2t_info = f"Epoch: {self.epoch} >>>  Video-to-Text:\n" \
                   f"R@1: {res['v2t_r1']:.3f} - R@5: {res['v2t_r5']:.3f} - R@10: {res['v2t_r10']:.3f}\n" \
                   f"Sum: {res['v2t_sum']:.3f} - Median R: {res['v2t_median']:.1f} - Mean R: {res['v2t_mean']:.1f}"

        my_log(t2v_info, logger='result')
        my_log(v2t_info, logger='result')
        res['t2v_info'] = t2v_info
        res['v2t_info'] = v2t_info
        return res

    def log_best(self):
        results = {
            't2v': sorted(self.results, key=lambda x: -x['t2v_r1'])[0],
            'v2t': sorted(self.results, key=lambda x: -x['v2t_r1'])[0],
            't2v_sum': sorted(self.results, key=lambda x: -x['t2v_sum'])[0],
            'v2t_sum': sorted(self.results, key=lambda x: -x['v2t_sum'])[0],
            'r1_sum': sorted(self.results, key=lambda x: -x['t2v_r1'] - x['v2t_r1'])[0],
            'r_sum': sorted(self.results, key=lambda x: -x['t2v_sum'] - x['v2t_sum'])[0],
        }
        for n, r in results.items():
            my_log(
                f"\n\nBest {n} Result: \n{r['t2v_info']}\n{r['v2t_info']}\n",
                logger='result'
            )

    def save_model(self, t2v_r1=0, v2t_r1=0):
        # Only save the model it-self
        model = self.model
        model_to_save = model.module if hasattr(model, 'module') else model
        postfix = "{}e_t2v{:0>4d}_v2t{:0>4d}.pth".format(
            self.epoch + 1, int(round(t2v_r1, 2) * 100), int(round(v2t_r1, 2) * 100), 
        )
        output_model_file = osp.join(self.work_dir, postfix)
        trainable_state_dict = { k: v for k, v in model_to_save.named_parameters() if v.requires_grad}
        torch.save(trainable_state_dict, output_model_file)
        my_log(f"Model saved to {output_model_file}", logger='result')
        return output_model_file


def get_rank(logits, prefix='t2v', label=None):
    if not isinstance(label, torch.Tensor): label = torch.arange(logits.shape[0])
    if isinstance(label, torch.Tensor) and len(label.shape) > 1:  # 矩阵
        all_logits = []
        for i, l in enumerate(label):
            cur_logits = logits[:, l > 0]
            all_logits.append(cur_logits.max(dim=-1, keepdim=True)[0])
        logits = torch.cat(all_logits, dim=1)
        label = torch.arange(logits.shape[0])

    # else:
    _, idx = logits.sort(dim=1, descending=True)    # 将logits降序，idx指向第大的索引
    _, rank = idx.sort(dim=1)       # 将idx升序排列，第i个指向第i大
    rank = rank[torch.arange(rank.shape[0]), label.long()]

    
    result = {
        '{}_r{}'.format(prefix, i): torch.sum(rank < i).item() / rank.shape[0] * 100
    for i in [1, 5, 10]}
    result['{}_sum'.format(prefix)] = sum(result.values())

    result['{}_mean'.format(prefix)] = rank.sum().item() / rank.shape[0]
    median_rank = rank.sort()[0].numpy().tolist()[rank.shape[0] // 2:(rank.shape[0] + 1) // 2 + 1]
    result['{}_median'.format(prefix)] = sum(median_rank) / len(median_rank)
    return result


def sec2timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    res = '{:0>2d}h {:0>2d}m {:0>2d}s'.format(h, m, s)
    return res
