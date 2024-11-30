#!/bin/bash

# Check if model, dataset, and GPU indices are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 \"model\" \"dataset\" \"lora_r\" \"n_nodes\" \"head_node_ip\" \"gpu_id_1 gpu_id_2 ...\""
    echo "Arg1: Model, Arg2: Dataset, Arg3: LoRA rank, Arg4: Number of nodes, Arg5: Head node IP, Arg6: GPU indices"
    echo "Example: $0 \"meditron-7b\" \"medqa\" \"16\" \"1\" \"127.0.0.1\" \"0 1 2 3\""
    exit 1
fi

model=$1
dataset=$2
let lora_r=$3
let n_nodes=$4
head_node_ip=$5

if [[ "$model" =~ "llama-3-8b" || "$model" == "openbiollm-8b" || "$model" == "med42-v2-8b" ]]; then
    if [[ "$dataset" == "pubmedqa" || "$dataset" == "medqa-usmle" ]]; then
        batch_size=8
    elif [[ "$dataset" == "ehrnoteqa" ]]; then
        batch_size=4
    elif [[ "$dataset" =~ "n2c2" ]]; then
        if [[ $lora_r -eq 64 ]]; then
            batch_size=1
        else
            batch_size=2
        fi
    elif [[ "$dataset" =~ "casi-sense" || "$dataset" =~ "mimic-iii-sense" ]]; then
        batch_size=8
    else
        batch_size=16
    fi
elif [[ "$dataset" == "pubmedqa" || "$dataset" == "casi-sense" ]]; then
    batch_size=16
elif [[ "$dataset" == "ehrnoteqa" || "$dataset" =~ "n2c2" ]]; then
    batch_size=4
else
    batch_size=32
fi

# Set up GPU devices
IFS=' ' read -r -a gpus <<< "$6"
IFS=',' gpus="${gpus[*]}"

# Run HPO
start_time=$(date +%s)
echo "Running FT HPO for ${model} on ${dataset}..."

log_dir="./llm_logs/${model}/${dataset}"
mkdir -p "${log_dir}"

export TOKENIZERS_PARALLELISM="false"
export MKL_SERVICE_FORCE_INTEL=1

if [[ $n_nodes -gt 1 ]]; then
    python3 -u ../../../src/llm/train_llm.py -m \
        model="${model}" \
        dataset="${dataset}" \
        n_nodes=$n_nodes \
        head_node_ip="${head_node_ip}" \
        gpu_ids="[${gpus}]" \
        training.per_device_train_batch_size=$batch_size \
        training.per_device_eval_batch_size=$batch_size \
        training.lora_r=$lora_r \
        training.learning_rate=1e-5,2e-5,5e-5,1e-4,2e-4 2>&1 | tee -a "${log_dir}/lora_r=${lora_r}.log"
else
    python3 -u ../../../src/llm/train_llm.py -m \
        model="${model}" \
        dataset="${dataset}" \
        gpu_ids="[${gpus}]" \
        training.per_device_train_batch_size=$batch_size \
        training.per_device_eval_batch_size=$batch_size \
        training.lora_r=$lora_r \
        training.learning_rate=1e-5,2e-5,5e-5,1e-4,2e-4 2>&1 | tee -a "${log_dir}/lora_r=${lora_r}.log"
fi
