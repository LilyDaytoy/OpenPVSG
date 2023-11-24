set -x

PARTITION=regular
JOB_NAME=psg
CONFIG=configs/mask2former_vps/mask2former_video_r50.py
WORK_DIR=work_dirs/mask2former_r50_vps
CHECKPOINT=work_dirs/mask2former_r50_vps/epoch_8.pth
PORT=${PORT:-$((29500 + $RANDOM % 29))}
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
    python -u tools/test.py ${CONFIG} ${CHECKPOINT} \
    --work-dir=${WORK_DIR} --eval PQ
