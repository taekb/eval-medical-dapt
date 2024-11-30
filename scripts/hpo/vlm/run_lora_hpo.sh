#!/bin/bash

# Check if model, dataset, lora_r, n_nodes, head_node_ip, and GPU indices are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 \"model\" \"dataset\" \"lora_r\" \"n_nodes\" \"head_node_ip\" \"gpu_id_1 gpu_id_2 ...\""
    echo "Arg1: Model, Arg2: Dataset, Arg3: LoRA rank, Arg4: Number of nodes, Arg5: Head node IP, Arg6: GPU indices"
    echo "Example: $0 \"llava-med-7b\" \"vqa-rad\" \"16\" \"1\" \"127.0.0.1\" \"0 1 2 3\""
    exit 1
fi

model=$1
dataset=$2
let lora_r=$3
let n_nodes=$4
head_node_ip=$5

# Set up GPU devices
IFS=' ' read -r -a gpus <<< "$6"
IFS=',' gpus="${gpus[*]}"

# Run HPO
start_time=$(date +%s)
echo "Running LoRA HPO for ${model} on ${dataset}..."

log_dir="./vlm_logs/${model}/${dataset}"
mkdir -p "${log_dir}"

if [[ $n_nodes -gt 1 ]]; then
    python3 -u ../../../src/vlm/train_vlm.py -m \
        model="${model}" \
        dataset="${dataset}" \
        n_nodes=$n_nodes \
        head_node_ip="${head_node_ip}" \
        gpu_ids="[${gpus}]" \
        training.bits=16 \
        training.lora_r=$lora_r \
        training.learning_rate=1e-5,2e-5,5e-5 \
        training.lora_dropout=0.,0.1 2>&1 | tee -a "${log_dir}/lora_r=${lora_r}.log"
else
    python3 -u ../../../src/vlm/train_vlm.py -m \
        model="${model}" \
        dataset="${dataset}" \
        gpu_ids="[${gpus}]" \
        training.bits=16 \
        training.lora_r=$lora_r \
        training.learning_rate=1e-5,2e-5,5e-5 \
        training.lora_dropout=0.,0.1 2>&1 | tee -a "${log_dir}/lora_r=${lora_r}.log"
fi