#!/bin/bash

# Check if model and dataset names are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 \"arg1\" \"arg1\" \"arg1\" \"arg1 arg2 ...\""
    echo "Arg1: Model, Arg2: Dataset, Arg3: Fine-tuning Method, Arg4: GPU indices"
    echo "Example: $0 \"llama-2-7b\" \"medqa\" \"lora\" \"0 1 2 3\""
    exit 1
fi

# Load base LLM in 4 bits if fine-tuned with QLoRA
if [[ $3 == "qlora" ]]; then
    load_in_4bit="true"
else
    load_in_4bit="false"
fi

# Read the provided arguments into arrays
model="${1}_${3}-${2}-best"
dataset="$2"
IFS=' ' read -r -a gpus <<< "$4"
IFS=',' gpus="${gpus[*]}"

# Run zero-shot inference
echo "Running zero-shot inference with ${model} on ${dataset}..."
python3 ../../../src/llm/infer_llm.py \
    model="${model}" \
    dataset="${dataset}" \
    gpu_ids="[${gpus}]" \
    gpu_memory_utilization=0.9 \
    prompt_type=zero-shot-ft \
    eval_method=exact-match \
    n_seeds=1 \
    force_eval=true \
    verbose=true \
    optimize_prompt=false \
    constrain_vocab=false \
    eval_split="test" \
    load_in_4bit="${load_in_4bit}"