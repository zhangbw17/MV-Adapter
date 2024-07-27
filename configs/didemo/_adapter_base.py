_base_ = '../retrieval_base_cfg_long.py'

work_dir='didemo/mv_adapter'

weight_decay = 0.2
optimizer = dict(
    trainable_type=['adapter.*', '.*_no_decay'],
    adapter_decay=dict(lr=5e-6, weight_decay=weight_decay),
    adapter_no_decay=dict(lr=5e-6),
    clip_decay=dict(weight_decay=weight_decay),
    other_decay=dict(lr=1e-4, weight_decay=weight_decay),
    other_no_decay=dict(lr=1e-4, weight_decay=weight_decay),
)

max_frames = 32
max_words = 64
train_dataset = dict(
    type='RetrievalDataset',
    data_root='data/didemo',
    ann='split/c4c_train.json',
    max_frames=max_frames
)
test_dataset = dict(
    type='RetrievalDataset',
    data_root='data/didemo',
    ann='split/c4c_test.json',
    max_frames=max_frames
)

model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=max_words,
)
