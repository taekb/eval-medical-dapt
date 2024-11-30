#!/bin/bash

# Check if model and dataset names are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg\" \"arg\""
    echo "Arg1: Models, Arg2: Datasets, Arg3: GPU indices, Arg4: Generation approach (one of \"greedy\"/\"logprob\"), Arg5: Medical model names"
    echo "Example: $0 \"llama-3-70b-instruct\" \"medqa medmcqa pubmedqa\" \"0 1 2 3\" \"greedy\" \"med42-v2-70b\""
    exit 1
fi

# Read the provided arguments into arrays
IFS=' ' read -r -a models <<< "$1"
IFS=' ' read -r -a datasets <<< "$2"
IFS=' ' read -r -a gpus <<< "$3"
IFS=',' gpus="${gpus[*]}"
IFS=' ' read -r -a med_model_names <<< "$5"

# Greedy decoding
if [[ $4 == "greedy" ]]; then
    n_seeds=1
    temperature=0
    constrain_vocab="false"

# Constrained log-probability prediction
elif [[ $4 == "logprob" ]]; then
    n_seeds=1
    temperature=0
    constrain_vocab="true"
else
    echo "Invalid generation approach. Choose one of \"greedy\"/\"logprob\"."
    exit 1
fi

# Set up number of few-shot examples to try
n_shots=(3)

for dataset in "${datasets[@]}"; do
    n_models=${#models[@]}

    for (( i=0; i<$n_models; i++ )); do
        model="${models[$i]}"
        med_model_name="${med_model_names[$i]}"

        # Check if we should skip zero-shot inference
        if [[ "$model" == "llama-2-7b" ]] && [[ "$dataset" == "ehrnoteqa" || "$dataset" == *"n2c2"* ]]; then
            echo "Skipping zero-shot inference for ${model} on ${dataset}."
            continue
        fi

        # Run zero-shot inference
        echo "Running zero-shot inference with ${model} on ${dataset}..."
        python3 -u ../../../src/llm/infer_llm.py \
            model="${model}" \
            dataset="${dataset}" \
            gpu_ids="[${gpus}]" \
            gpu_memory_utilization=0.9 \
            prompt_type=zero-shot \
            eval_method=exact-match \
            n_seeds="${n_seeds}" \
            temperature="${temperature}" \
            force_eval=false \
            verbose=true \
            optimize_prompt=false \
            constrain_vocab="${constrain_vocab}" \
            use_med_prompt=true \
            med_model_name="${med_model_name}"
        
        # Run few-shot inference
        if [[ "$dataset" != "ehrnoteqa" && "$dataset" != *"n2c2"* ]]; then
            for n in "${n_shots[@]}"; do
                echo "Running ${n}-shot inference with ${model} on ${dataset}..."
                python3 -u ../../../src/llm/infer_llm.py \
                    model="${model}" \
                    dataset="${dataset}" \
                    gpu_ids="[${gpus}]" \
                    gpu_memory_utilization=0.9 \
                    prompt_type=few-shot \
                    n_shot="${n}" \
                    eval_method=exact-match \
                    n_seeds="${n_seeds}" \
                    temperature="${temperature}" \
                    force_eval=false \
                    verbose=true \
                    optimize_prompt=false \
                    constrain_vocab="${constrain_vocab}" \
                    use_med_prompt=true \
                    med_model_name="${med_model_name}"
            done
        fi
    done
done