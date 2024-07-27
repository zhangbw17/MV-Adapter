
weight_decay = 0.2
optimizer = dict(
    trainable_type=['.*'],
    warmup_proportion=0.1,

    default=dict(lr=1e-7, weight_decay=0),

    adapter_decay=dict(lr=5e-6, weight_decay=weight_decay),
    adapter_no_decay=dict(lr=5e-6, weight_decay=0),
    clip_decay=dict(lr=1e-7, weight_decay=weight_decay),
    clip_no_decay=dict(lr=1e-7, weight_decay=weight_decay),
    other_decay=dict(lr=1e-4, weight_decay=weight_decay),
    other_no_decay=dict(lr=1e-4, weight_decay=weight_decay),
)

train_cfg = dict(
    optimizer=optimizer,
    max_epochs=5, val_begin=0, val_interval=1,
)

data_modal = 'image'

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    pin_memory=True,
    shuffle=False,
    drop_last=True,
    multiprocessing_context='fork'
)
val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    multiprocessing_context='fork'
)

test_cfg = None

checkpoint = None

log_step = 50
tau = 'constant'
seed  = 0
work_dir='./log'
