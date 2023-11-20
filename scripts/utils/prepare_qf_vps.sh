#!/usr/bin/env bash
set -x

PARTITION=priority
JOB_NAME=vps_feat
CONFIG=configs/mask2former_vps/mask2former_video_r50_single_video_test.py
CHECKPOINT=work_dirs/mask2former_r50_vps_1108/epoch_8.pth
SPLIT=train
WORK_DIR=work_dirs/vps_${SPLIT}_save_qf_1115
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
    python -u tools/prepare_query_tube_vps.py ${CONFIG} ${CHECKPOINT} \
    --work-dir ${WORK_DIR} --split ${SPLIT} --launcher="none" ${PY_ARGS}

# sh scripts/utils/prepare_qf_vps.sh
