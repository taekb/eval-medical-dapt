#!/bin/bash

# Check if GPU indices and generation approach were specified
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg1 arg2 ...\" \"arg\" \"arg\" \"arg\""
    echo "Arg1: Models, Arg2: Datasets, Arg3: Fine-tuning Method (\"ft\"/\"lora\"/\"qlora\"), Arg4: Number of GPUs Used for Training"
    echo "Example: $0 \"llama-3-8b openbiollm-8b\" \"medqa medmcqa pubmedqa\" \"lora\" \"2\""
    exit 1
fi

IFS=' ' read -r -a models <<< "$1"
IFS=' ' read -r -a datasets <<< "$2"
ft_method=$3
let n_gpus=$4

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # For (Q)LoRA
        if [[ $ft_method == "lora" || $ft_method == "qlora" ]]; then
            
            if [[ $model =~ "llama-3-8b" || $model =~ "openbiollm-8b" || $model =~ "med42-v2-8b" ]]; then
                if [[ $dataset == "pubmedqa" || $dataset == "medqa-usmle" ]]; then
                    batch_size=$(( 8 * n_gpus ))
                elif [[ $dataset == "ehrnoteqa" ]]; then
                    batch_size=$(( 4 * n_gpus ))
                elif [[ $dataset =~ "n2c2" ]]; then
                    batch_size=$(( 2 * n_gpus ))
                elif [[ $dataset =~ "casi-sense" || $dataset =~ "mimic-iii-sense" ]]; then
                    batch_size=$(( 8 * n_gpus ))
                else
                    batch_size=$(( 16 * n_gpus ))
                fi
            elif [[ $dataset == "pubmedqa" || $dataset == "casi-sense" ]]; then
                batch_size=$(( 16 * n_gpus ))
            elif [[ $dataset == "ehrnoteqa" || $dataset =~ "n2c2" ]]; then
                batch_size=$(( 4 * n_gpus ))
            else 
                batch_size=$(( 32 * n_gpus ))
            fi
            
            python3 ../../../src/find_best_model.py \
                --models "${model}" \
                --model_type "llm" \
                --datasets "${dataset}" \
                --ft_method "${ft_method}" \
                --epochs 10 \
                --lr 1e-5 2e-5 5e-5 1e-4 2e-4 \
                --scheduler cosine \
                --batch_size "${batch_size}" \
                --decay 0.0 \
                --warmup 0.05 \
                --lora_r 16 32 64 \
                --lora_alpha 16 \
                --lora_dropout 0.0 \
                --lora_bias none
        
        else
            echo "Full fine-tuning not considered in main experiments."
        fi
    done
done

