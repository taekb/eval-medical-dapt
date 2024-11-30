#!/bin/bash

# Check if GPU indices and generation approach were specified
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg\" \"arg\""
    echo "Arg1: Models, Arg2: Datasets, Arg3: Fine-tuning Method (\"ft\"/\"lora\"/\"qlora\")"
    echo "Example: $0 \"llava-med-7b llava-v0-7b\" \"vqa-rad pvqa slake\" \"lora\""
    exit 1
fi

IFS=' ' read -r -a models <<< "$1"
IFS=' ' read -r -a datasets <<< "$2"
ft_method=$3

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # For LLaVA models
        if [[ $model =~ "llava" ]]; then
            python3 ../../../src/find_best_model.py \
                --models "${model}" \
                --model_type "vlm" \
                --datasets "${dataset}" \
                --ft_method "${ft_method}" \
                --epochs 10 \
                --lr 1e-5 2e-5 5e-5 \
                --scheduler cosine \
                --batch_size 64 \
                --decay 0.0 \
                --warmup 0.03 \
                --lora_r 16 32 64 \
                --lora_alpha 16 \
                --lora_dropout 0.0 \
                --lora_bias none \
                --freeze_mm True \
                --mm_lr 2e-5
        
        # For OpenFlamingo models
        elif [[ $model =~ "flamingo" ]]; then
            python3 ../../../src/find_best_model.py \
                --models "${model}" \
                --model_type "vlm" \
                --datasets "${dataset}" \
                --ft_method "${ft_method}" \
                --epochs 10 \
                --lr 1e-5 2e-5 5e-5 \
                --scheduler cosine \
                --batch_size 16 \
                --decay 0.0 0.05 0.1 \
                --warmup 0.05 \
                --xattn 4
        fi
    done
done
