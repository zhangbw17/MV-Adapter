_base_ = '../retrieval_base_cfg.py'

work_dir='lsmdc/c4c'

weight_decay = 0.2
optimizer = dict(
    adapter_decay=dict(lr=5e-6, weight_decay=weight_decay),
    adapter_no_decay=dict(lr=5e-6),
    clip_decay=dict(weight_decay=weight_decay),
    other_decay=dict(lr=1e-4, weight_decay=weight_decay),
    other_no_decay=dict(lr=1e-4, weight_decay=weight_decay),
)

train_cfg = dict(
    optimizer=optimizer,
    max_epochs=5, val_begin=0, val_interval=1,
)

max_frames = 12
max_words = 32
train_dataset = dict(
    type='RetrievalDataset',
    data_root='data/lsmdc',
    ann='split/c4c_train.json',
)
test_dataset = dict(
    type='RetrievalDataset',
    data_root='data/lsmdc',
    ann='split/c4c_test.json',
)


model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=max_words,
)
