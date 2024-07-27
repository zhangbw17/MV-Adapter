from abc import ABC, abstractmethod


class Runner(ABC):
    def __init__(
        self,
        cfg,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,

    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.epoch = 0
        self.global_step = 0
        self.work_dir = cfg.work_dir
        self.max_epochs = cfg.train_cfg.max_epochs
        self.max_steps = cfg.total_step
    
    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def train(self):
        pass
