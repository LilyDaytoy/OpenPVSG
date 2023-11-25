#!/usr/bin/env bash
set -x

PARTITION=priority
JOB_NAME=tracking
CONFIG=configs/unitrack/imagenet_resnet50_s3_womotion_timecycle.py
CHECKPOINT=work_dirs/mask2former_r50_ips/epoch_8.pth
SPLIT=val
WORK_DIR=work_dirs/${SPLIT}_save
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}

PYTHONPATH="/mnt/lustre/jkyang/CVPR23/openpvsg":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/prepare_query_tube.py ${CONFIG} ${CHECKPOINT} \
    --work-dir ${WORK_DIR} --split ${SPLIT} --launcher="none" ${PY_ARGS}
