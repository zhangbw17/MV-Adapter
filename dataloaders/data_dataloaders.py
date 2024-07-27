import torch
from torch.utils.data import DataLoader

from registry import DATASETS


collate_func = dict(
    stack=torch.stack,
    cat=torch.cat,
    list_plus=lambda x : x,
    tensor=lambda x: torch.tensor(x),
)

def collate_fn(batch_data):
    collated_data = dict()
    for func in batch_data[0]:
        c_func = collate_func[func]
        cur_data = {k: c_func([bd[func][k] for bd in batch_data]) for k in batch_data[0][func]}
        collated_data.update(cur_data)
    return collated_data


def build_loader(cfg):
    train_dt_cls = cfg.train_dataset.pop('type', 'RetrievalDataset')
    test_dt_cls = cfg.test_dataset.pop('type', 'RetrievalDataset')
    train_dataset = DATASETS.get(train_dt_cls)(**cfg.train_dataset)
    test_dataset = DATASETS.get(test_dt_cls)(**cfg.test_dataset)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        **cfg.train_dataloader,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        collate_fn=collate_fn,
        **cfg.val_dataloader
    )

    train_steps = int(len(train_dataloader)) * cfg.train_cfg.max_epochs
    cfg.total_step = train_steps

    return train_dataloader, val_dataloader
