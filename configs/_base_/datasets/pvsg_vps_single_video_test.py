dataset_type = 'PVSGVideoSingleVideoDataset'
data_root = './data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False
)
crop_size = (360, 480)


test_pipeline = [
    dict(type='LoadMultiImagesDirect'),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='VideoCollect', keys=['img']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='test',
        ref_seq_len_test=1,
        ref_seq_index=None,
        test_mode=True,
        pipeline=test_pipeline,
        video_name="0010_8610561401", # for a single video
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',
        ref_sample_mode='sequence',
        ref_seq_index=[0, 1],
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

evaluation = dict(interval=5000000)
