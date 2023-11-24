#!/bin/bash
set -x

# sh scripts/train/train_relation_loop.sh
# Define the arrays for the parameters
ps_types=("ips")
model_names=("vanilla" "filter" "conv" "transformer")

# Default values
PARTITION=priority
JOB_NAME=psg
PORT=${PORT:-$((29500 + $RANDOM % 29))}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:5}

# Iterate over the arrays and run each combination
for ps_type in "${ps_types[@]}"; do
    for model_name in "${model_names[@]}"; do
        PYTHONPATH="/mnt/lustre/jkyang/CVPR23/openpvsg":$PYTHONPATH \
        srun -p ${PARTITION} \
            --job-name=${JOB_NAME}_${ps_type}_${model_name} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks-per-node=${GPUS_PER_NODE} \
            --cpus-per-task=${CPUS_PER_TASK} \
            --kill-on-bad-exit=1 \
            python -u tools/rel_train.py --ps-type ${ps_type} --model-name ${model_name} &
    done
done

# Wait for all background processes to finish
wait
