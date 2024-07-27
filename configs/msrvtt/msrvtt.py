_base_ = './_adapter_base.py'

work_dir='msrvtt'

max_frames = 12
max_words = 32

model = dict(
    type='BaselineModel',
    clip_cache_dir='~/.cache/clip',

    max_words=max_words,
    adapter=dict(
        modal_dynamic=dict(
            adapter_idx=list(range(10, 12)),
            factor_dim=32,
            position='down out'
        ),
        visual=dict(
            mlp=dict(
                adapter_idx=list(range(12)),
                forward='seq',
                type='VideoAdapter',
                embed_dim=768,
                cls_mid=64,
                n_head=2,
                seq_len=max_frames,
                temporal=['c'],
                pca=False,
            )
        ),
        text=dict(
            mlp=dict(
                adapter_idx=list(range(12)),
                forward='seq',
                type='TextTransfAdapter',
                embed_dim=512,
                mid_dim=64,
                n_head=2,
                attn_type='uni',
                seq_len=max_words
            )
        ),
    )
)
