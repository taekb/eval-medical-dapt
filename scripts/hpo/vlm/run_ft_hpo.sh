#!/bin/bash

# Check if model, dataset, and GPU indices are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 \"model\" \"dataset\" \"gpu_id_1 gpu_id_2 ...\""
    echo "Arg1: Model, Arg2: Dataset, Arg3: GPU indices"
    echo "Example: $0 \"open-flamingo-9b\" \"vqa-rad\" \"0 1 2 3\""
    exit 1
fi

model=$1
dataset=$2

if [[ $model =~ "llava" ]]; then
    training="llava"
elif [[ $model =~ "flamingo" ]]; then
    training="flamingo"
fi

# Set up GPU devices
IFS=' ' read -r -a gpus <<< "$3"
IFS=',' gpus="${gpus[*]}"

# Run HPO
start_time=$(date +%s)
echo "Running FT HPO for ${model} on ${dataset}..."

log_dir="./vlm_logs/${model}/${dataset}"
mkdir -p "${log_dir}"

python3 -u ../../../src/vlm/train_vlm.py -m \
    model="${model}" \
    dataset="${dataset}" \
    training="${training}" \
    gpu_ids="[${gpus}]" \
    training.learning_rate=1e-5,2e-5,5e-5 \
    training.weight_decay=0.,0.05,0.1 2>&1 | tee -a "${log_dir}/ft.log"