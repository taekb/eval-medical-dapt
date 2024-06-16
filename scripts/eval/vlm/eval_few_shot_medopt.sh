#!/bin/bash

# Check if model and dataset names are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg\" \"arg\""
    echo "Arg1: Models, Arg2: Datasets, Arg3: GPU indices, Arg4: Generation approach (one of \"greedy\"/\"sc\"/\"logprob\")"
    echo "Example: $0 \"llava-med-7b llava-v0-7b\" \"vqa-rad pvqa slake\" \"0 1 2 3\" \"greedy\""
    exit 1
fi

# Read the provided arguments into arrays
IFS=' ' read -r -a models <<< "$1"
IFS=' ' read -r -a datasets <<< "$2"
IFS=' ' read -r -a gpus <<< "$3"
IFS=',' gpus="${gpus[*]}"

# Greedy decoding
if [[ $4 == "greedy" ]]; then
    n_seeds=1
    predict_with_logprob="false"

# Self-consistency decoding
elif [[ $4 == "sc" ]]; then
    n_seeds=5
    predict_with_logprob="false"

# Top-token prediction
elif [[ $4 == "logprob" ]]; then
    n_seeds=1
    predict_with_logprob="true"
else
    echo "Invalid generation approach. Choose one of \"greedy\"/\"sc\"/\"logprob\"."
    exit 1
fi

# Set up number of few-shot examples to try
n_shots=(3)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Run zero-shot inference
        echo "Running zero-shot inference with ${model} on ${dataset}..."
        python3 -u ../../../src_backup/vlm/infer_vlm.py \
            model="${model}" \
            dataset="${dataset}" \
            gpu_ids="[${gpus}]" \
            prompt_type=zero-shot \
            load_in_4bit=false \
            n_seeds="${n_seeds}" \
            force_eval=true \
            verbose=true \
            optimize_prompt=false \
            use_med_prompt=true \
            predict_with_logprob="${predict_with_logprob}"
        
        # Run few-shot inference if the model hasn't been fine-tuned
        for n in "${n_shots[@]}"; do
            echo "Running ${n}-shot inference with ${model} on ${dataset}..."
            python3 -u ../../../src_backup/vlm/infer_vlm.py \
                model="${model}" \
                dataset="${dataset}" \
                gpu_ids="[${gpus}]" \
                prompt_type=few-shot \
                load_in_4bit=false \
                n_seeds="${n_seeds}" \
                n_shot="${n}" \
                force_eval=true \
                verbose=true \
                optimize_prompt=false \
                use_med_prompt=true \
                predict_with_logprob="${predict_with_logprob}"
        done
    done
done
