#!/bin/bash

# Check if GPU indices were specified
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg\" \"arg\""
    echo "Arg1: GPU indices, Arg2: Generation approach (one of \"greedy\"/\"logprob\"), Arg3: Optimize prompts (\"true\"/\"false\")"
    echo "Example: $0 \"0 1 2 3\" \"greedy\" \"true\""
    exit 1
fi

datasets="vqa-rad pvqa slake mmmu_basic-medical-science mmmu_clinical-medicine mmmu_diagnostics-and-laboratory-medicine mmmu_pharmacy mmmu_public-health"

echo "Running comparison of few-shot performance between medical and general-domain models..."

start_time=$(date +%s)
logdir="./vlm_logs/compare_medical_general"
mkdir -p "${logdir}"

eval "$(conda shell.bash hook)"

# LLaVA-Med-7B and LLaVA-v0-7B
conda activate llava-env
./eval_few_shot.sh "llava-med-7b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llava-med_${2}.log"
./eval_few_shot.sh "llava-v0-7b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llava-v0_${2}.log" 

# Med-Flamingo-9B and Open-Flamingo-9B
conda activate open-flamingo-env
./eval_few_shot.sh "med-flamingo-9b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/med-flamingo_${2}.log"
./eval_few_shot.sh "open-flamingo-9b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/open-flamingo_${2}.log"

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"