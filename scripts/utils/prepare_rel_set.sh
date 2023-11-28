set -x

# sh scripts/utils/prepare_rel_set.sh

PARTITION=priority
JOB_NAME=psg
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SPLIT=val
WORK_DIR=work_dirs/ips_${SPLIT}_save_qf

PYTHONPATH="/mnt/lustre/jkyang/CVPR23/openpvsg":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python tools/prepare_rel_set_dist.py \
    --split ${SPLIT} --work_dir ${WORK_DIR}
