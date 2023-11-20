_base_ = [
    '../_base_/datasets/pvsg_vps_single_video_test.py',
    './mask2former_video_r50_base.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/m2f_schedules.py',
]

# load mask2former coco r50
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/' \
            'mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic/' \
            'mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth'

dataset_type = 'PVSGVideoSingleVideoDataset'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            split='train',
            ref_sample_mode='sequence',
            ref_seq_index=[0,1],
            test_mode=False,
        )
    ),
    test=dict(  # need test set for cfg_compact 
        type=dataset_type,
        split='val',
        video_name="0010_8610561401", # for a single video
        ref_sample_mode='test',
        ref_seq_len_test=1,
        ref_seq_index=None,
        test_mode=True,
    )
)

model = dict(
    panoptic_head=dict(
        loss_sem_seg=None
    ),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHeadCustom',
        loss_panoptic=None,
        init_cfg=None,
    ),

    train_cfg=dict(
        num_points=12544,
    ),
    test_cfg=dict(
        object_mask_thr=0.8,
        iou_thr=0.8,
        filter_low_score=True,
        return_query=True,
    ),
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7,]
)
runner = dict(type='EpochBasedRunner', max_epochs=8)

project_name = 'pvsg'
expt_name = 'vps_1108'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook', 
        init_kwargs=dict(
            project=project_name,
            name=expt_name,
        ),
        )
    ])

