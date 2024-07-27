_base_ = './base.py'


weight_decay = 0.2
optimizer = dict(
    trainable_type=['adapter*'],
    adapter_decay=dict(lr=5e-6, weight_decay=weight_decay),
    adapter_no_decay=dict(lr=5e-6),
    clip_decay=dict(weight_decay=weight_decay),
    other_decay=dict(lr=1e-4, weight_decay=weight_decay),
    other_no_decay=dict(lr=1e-4, weight_decay=weight_decay),
)

train_cfg = dict(
    optimizer=optimizer,
    max_epochs=5, val_begin=1, val_interval=1,
)

model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',
    max_words=32,
    adapter=dict(
        visual=dict(
            adapter_idx=list(range(12)),
            mlp=dict(
                forward='seq',
                type='VideoAdapter',
                embed_dim=768,
                cls_mid=64,
                conv_mid=16,
                n_head=2,
            )
        ),
        text=dict(
            adapter_idx=list(range(12)),
            mlp=dict(
                forward='seq',
                type='TextTransfAdapter',
                embed_dim=512,
                cls_mid=64,
                conv_mid=16,
                n_head=2,
                attn_type='uni',
                seq_len=32
            )
        ),
    )
)
