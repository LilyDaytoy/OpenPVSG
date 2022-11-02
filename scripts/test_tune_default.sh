#!/usr/bin/env bash

set -x

PARTITION=dsta
JOB_NAME=test_tune_default
CONFIG=configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic_custom.py
CHECKPOINT=/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image/work_dirs/pvsg_demo_mask2former_r50_default_train_slurm/epoch_4.pth
WORK_DIR=work_dirs/test_tune_default_with_query_ckpt4
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}
node=68
OUT=work_dirs/test_tune_default_with_query_ckpt4/outs.pickle

PYTHONPATH="$(/mnt/lustre/jkyang/wxpeng/CVPR23/PVSG_Image $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-${node} \
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} --out ${OUT} --work-dir ${WORK_DIR} --eval "True" --launcher="slurm" ${PY_ARGS}
