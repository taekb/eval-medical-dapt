#!/bin/bash

# Check if model and dataset names are provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg\", \"arg\""
    echo "Arg1: Models, Arg2: Datasets, Arg3: GPU indices, Arg4: Generation approach (one of \"greedy\"/\"logprob\"), Arg5: Optimize prompts (\"true\"/\"false\")"
    echo "Example: $0 \"mistral-7b-v0.1 biomistral-7b\" \"medqa medmcqa pubmedqa\" \"0 1 2 3\" \"greedy\" \"true\""
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

# Option to optimize prompts
if [[ $5 == "true" ]]; then
    optimize_prompt="true"
else
    optimize_prompt="false"
fi

# Set up number of few-shot examples to try
n_shots=(3)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        # Check if we should skip zero-shot inference
        if [[ "$model" == "llama-2-7b" || "$model" == "meditron-7b" ]] && \
            [[ "$dataset" == "ehrnoteqa" || "$dataset" == *"n2c2"* ]]; then
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
            optimize_prompt="${optimize_prompt}" \
            constrain_vocab="${constrain_vocab}"

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
                    optimize_prompt="${optimize_prompt}" \
                    constrain_vocab="${constrain_vocab}"
            done
        else
            echo "Skipping few-shot evaluation for EHRNoteQA / n2c2 datasets."
        fi
    done
done