#!/bin/bash
set -x

# sh scripts/test/test_relation_full_loop.sh
# Define the arrays for the parameters
work_dirs=(
    "./work_dirs/relation/rel_ips_vanilla"
    "./work_dirs/relation/rel_ips_filter"
    "./work_dirs/relation/rel_ips_conv"
    "./work_dirs/relation/rel_ips_transformer"
)
epoch_list=(100)

# Default values
PARTITION=priority
JOB_NAME=psg
PORT=${PORT:-$((29500 + $RANDOM % 29))}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}

# Iterate over the arrays and run each combination
for work_dir in "${work_dirs[@]}"; do
    for epoch_id in "${epoch_list[@]}"; do
        PYTHONPATH="/mnt/lustre/jkyang/CVPR23/openpvsg":$PYTHONPATH \
        srun -p ${PARTITION} \
            --job-name=${JOB_NAME}_${ps_type}_${model_name} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks-per-node=${GPUS_PER_NODE} \
            --cpus-per-task=${CPUS_PER_TASK} \
            --quotatype auto \
            --kill-on-bad-exit=1 \
            python -u tools/rel_test_full.py \
            --work-dir ${work_dir} --epoch-id ${epoch_id}&
    done
done

# Wait for all background processes to finish
wait
