#!/bin/bash

# Check if GPU indices and generation approach were specified
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 \"arg1 arg2 ...\" \"arg\" \"arg\""
    echo "Arg1: GPU indices, Arg2: Generation approach (one of \"greedy\"/\"logprob\"), Arg3: Optimize prompts (\"true\"/\"false\")"
    echo "Example: $0 \"0 1 2 3\" \"greedy\" \"true\""
    exit 1
fi

datasets="medqa medmcqa pubmedqa mmlu_anatomy mmlu_clinical-knowledge mmlu_college-biology mmlu_college-medicine mmlu_medical-genetics mmlu_professional-medicine mmlu_high-school-biology mmlu_virology mmlu_nutrition mednli ehrnoteqa n2c2_2008-obesity_asthma n2c2_2008-obesity_cad n2c2_2008-obesity_diabetes n2c2_2008-obesity_obesity casi-sense mimic-iii-sense"

echo "Running comparison of few-shot performance between medical and general-domain models..."

start_time=$(date +%s)
logdir="./llm_logs/compare_medical_general"
mkdir -p "${logdir}"

# Llama-3-70B-Instruct vs. OpenBioLLM-70B & Med42-v2-70B
./eval_few_shot.sh "llama-3-70b-instruct" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llama-3-70b-instruct_${2}.log"
./eval_few_shot.sh "openbiollm-70b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/openbiollm-70b_${2}.log"
./eval_few_shot.sh "med42-v2-70b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/med42-v2-70b_${2}.log"

# Llama-3-8B-Instruct vs. Med42-v2-8B
./eval_few_shot.sh "llama-3-8b-instruct" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llama-3-8b-instruct_${2}.log"
./eval_few_shot.sh "med42-v2-8b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/med42-v2-8b_${2}.log"

# Llama-3-8B vs. OpenBioLLM-8B
./eval_few_shot.sh "llama-3-8b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llama-3-8b_${2}.log"
./eval_few_shot.sh "openbiollm-8b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/openbiollm-8b_${2}.log"

# Mistral-7B-Instruct-v0.1 vs. BioMistral-7B
./eval_few_shot.sh "mistral-7b-v0.1" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/mistral-7b-v0.1_${2}.log"
./eval_few_shot.sh "biomistral-7b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/biomistral-7b_${2}.log"

# Llama-2-7B-Chat vs. BioMedGPT-7B
./eval_few_shot.sh "llama-2-7b-chat" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llama-2-7b-chat_${2}.log"
./eval_few_shot.sh "biomedgpt-7b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/biomedgpt-7b_${2}.log"

# Llama-2-70B vs. MediTron-70B & Clinical-Camel-70B & Med42-v1-70B
./eval_few_shot.sh "llama-2-70b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llama-2-70b_${2}.log"
./eval_few_shot.sh "meditron-70b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/meditron-70b_${2}.log"
./eval_few_shot.sh "clinical-camel-70b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/clinical-camel-70b_${2}.log"
./eval_few_shot.sh "med42-v1-70b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/med42-v1-70b_${2}.log"

# Llama-2-7B vs. MediTron-7B
./eval_few_shot.sh "llama-2-7b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/llama-2-7b_${2}.log"
./eval_few_shot.sh "meditron-7b" "${datasets}" "$1" "$2" "$3" 2>&1 | tee -a "${logdir}/meditron-7b_${2}.log"

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

printf "\nTotal elapsed time: ${hours} hours, ${minutes} minutes, ${seconds} seconds.\n\n"