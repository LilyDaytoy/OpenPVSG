set -x
# sh scripts/utils/show_results.sh
PARTITION=priority
JOB_NAME=psg
PORT=${PORT:-$((29500 + $RANDOM % 29))}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

PYTHONPATH="/mnt/lustre/jkyang/CVPR23/openpvsg":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype auto \
    --kill-on-bad-exit=1 \
    python tools/show_pred_results.py
