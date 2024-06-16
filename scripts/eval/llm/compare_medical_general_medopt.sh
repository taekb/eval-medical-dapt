#!/bin/bash

# Check if GPU indices and generation approach were specified
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg\" "
    echo "Arg1: GPU indices, Arg2: Generation approach (one of \"greedy\"/\"sc\"/\"logprob\")"
    echo "Example: $0 \"0 1 2 3\" \"greedy\""
    exit 1
fi

datasets="medqa medmcqa pubmedqa mmlu_anatomy mmlu_clinical-knowledge mmlu_college-biology mmlu_college-medicine mmlu_medical-genetics mmlu_professional-medicine mmlu_high-school-biology mmlu_virology mmlu_nutrition"

echo "Running comparison of few-shot performance between medical and general-domain models..."

start_time=$(date +%s)
logdir="./llm_logs/compare_medical_general_medopt"
mkdir -p "${logdir}"

# Llama-3-70B-Instruct
./eval_few_shot_medopt.sh "llama-3-70b" "${datasets}" "$1" "$2" 2>&1 | tee -a "${logdir}/llama-3-70b_${2}.log"

# Llama-3-8B-Instruct
./eval_few_shot_medopt.sh "llama-3-8b" "${datasets}" "$1" "$2" 2>&1 | tee -a "${logdir}/llama-3-8b_${2}.log"

# Mistral-7B-v0.1
./eval_few_shot_medopt.sh "mistral-7b-v0.1" "${datasets}" "$1" "$2" 2>&1 | tee -a "${logdir}/mistral-7b-v0.1_${2}.log"

# Llama-2-7B-Chat
./eval_few_shot_medopt.sh "llama-2-7b-chat" "${datasets}" "$1" "$2" 2>&1 | tee -a "${logdir}/llama-2-7b-chat_${2}.log"

# Llama-2-70B
./eval_few_shot_medopt.sh "llama-2-70b" "${datasets}" "$1" "$2" 2>&1 | tee -a "${logdir}/llama-2-70b_${2}.log"

# Llama-2-7B
./eval_few_shot_medopt.sh "llama-2-7b" "${datasets}" "$1" "$2" 2>&1 | tee -a "${logdir}/llama-2-7b_${2}.log"

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"