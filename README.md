# Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress?

This is the source code repository for the paper: "Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress?"

We include all of the code used for preprocessing the medical QA datasets and running the main zero-/few-shot prompting experiments discussed in Section 3 and 4. For all medical and general-domain LLMs and VLMs used for evaluation, we use the HuggingFace checkpoints listed below.

LLMs
- OpenBioLLM-70B: `aaditya/Llama3-OpenBioLLM-70B`
- Llama-3-70B-Instruct: `meta-llama/Meta-Llama-3-70B-Instruct`
- MediTron-70B: `epfl-llm/meditron-70b`
- Llama-2-70B: `meta-llama/Llama-2-70b-hf`
- OpenBioLLM-8B: `aaditya/Llama3-OpenBioLLM-8B`
- Llama-3-8B: `meta-llama/Meta-Llama-3-8B`
- MediTron-7B: `epfl-llm/meditron-7b`
- Llama-2-7B: `meta-llama/Llama-2-7b-hf`
- BioMistral-7B: `BioMistral/BioMistral-7B`
- Mistral-7B-Instruct-v0.1: `mistralai/Mistral-7B-Instruct-v0.1`
- BioMedGPT-LM-7B: `PharMolix/BioMedGPT-LM-7B`
- Llama-2-7B-Chat: `meta-llama/Llama-2-7b-chat-hf`

VLMs
- LLaVA-Med-7B: `microsoft/llava-med-7b-delta`
- LLaVA-v0-7B: `liuhaotian/LLaVA-7b-delta-v0`
- Med-Flamingo-9B: `med-flamingo/med-flamingo`
- Open-Flamingo-9B: `openflamingo/OpenFlamingo-9B-deprecated`

For LLaVA-Med-7B and LLaVA-v0-7B, we note that the checkpoints provided are delta weights that cannot be used directly. Please see the instructions provided in the [LLaVA-Med repository](https://github.com/microsoft/LLaVA-Med/tree/b9a98a736d2ef05bcf5ff345be6403fb3a664eaf?tab=readme-ov-file#training) and the [LLaVA repository](https://github.com/haotian-liu/LLaVA) for merging the delta weights with the base LLaMA-7B LLM weights: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#legacy-models-delta-weights.

<br>

## 1. Setup
To set up the conda environment (`llm-env`) that we used all of the LLM experiments, run the following:
```
./scripts/setup/setup_llm.sh
```

To set up the conda environment (`llava-env`) that we used for all of the experiments with LLaVA-Med-7B and LLaVA-v0-7B, run the following:
```
./scripts/setup/setup_llava.sh
```

To set up the conda environment (`open-flamingo-env`) that we used for all of the experiments with Med-Flamingo-9B and Open-Flamingo-9B, run the following:
```
./scripts/setup/setup_flamingo.sh
```

## 2. Data Loading and Preprocessing

### 2.1 Textual Medical QA Datasets
For all of the textual medical QA datasets, instantiating the relevant dataset class (e.g., `MedQADataset`, `MMLUMedicalDataset`) in `./src/dataset.py` will automatically download and cache the data to the path specified by the `hf_cache_dir` argument:
```
dataset = MedQADataset(
    splits=['train', 'test'], 
    main_split='test',
    hf_cache_dir='/data'
)
```
For zero-shot prompting, running the following will apply a prompt template specified in the argument to all of the QA examples in the `main_split` in the zero-shot format (i.e., system prompt + question):
```
dataset.load_and_apply_prompt_template(
    model_name='llama-3-8b', # Use the default prompt format for Llama-3-8B
    prompt_type='zero-shot', # Zero-shot prompting format
    tokenizer=tokenizer # Assuming model tokenizer has been loaded beforehand
)
```
To randomly sample a prompt format using the context-free grammar we show in Section 3 and Appendix B, you can additionally pass the `sample_kwargs' argument to, with the desired fixed random seeds.
```
dataset.load_and_apply_prompt_template(
    model_name='llama-3-8b', # Use the default prompt format for Llama-3-8B
`   sample_kwargs=dict(prompt_template_seed=0)
    prompt_type='zero-shot', 
    tokenizer=tokenizer 
)
```
For few-shot prompting, call the `sample_few_shot_qas()` method before calling `load_and_apply_prompt_template()`:
```
dataset = MedQADataset(
    splits=['train', 'test'], 
    main_split='test',
    hf_cache_dir='/data'
)
dataset.sample_few_shot_qas(n_shot=3, seed=0)
dataset.load_and_apply_prompt_template(
    model_name='llama-3-8b',
`   sample_kwargs=dict(prompt_template_seed=0)
    prompt_type='few-shot', 
    tokenizer=tokenizer 
)
```

### 2.2. Visual Medical QA Datasets
The MMMU-Medical datasets can be directly loaded from HuggingFace, as with all of the textual medical QA datasets. Below is an example for loading the MMMU (Basic Medical Science) dataset for 3-shot prompting LLaVA-Med-7B:
```
dataset = MMMUDataset(
    name='mmmu_basic-medical-science',
    splits=['train', 'test'], 
    main_split='test',
    hf_cache_dir='/data'
)
dataset.sample_few_shot_qas(n_shot=3, seed=0)
dataset.load_and_apply_prompt_template(
    model_name='llava-med-7b',
`   sample_kwargs=dict(prompt_template_seed=0)
    prompt_type='few-shot', 
    tokenizer=tokenizer 
)
```

All other datasets should be downloaded separately from the official repositories beforehand: [VQA-RAD](https://osf.io/89kps/), [PathVQA](https://github.com/UCSD-AI4H/PathVQA), [SLAKE](https://www.med-vqa.com/slake/). For these datasets, which contain both closed-ended and open-ended QA examples, we performed additional preprocessing to only extract the closed-ended QA examples and format them into structured `.jsonl` files, as detailed in `./notebooks/preprocess-vqa.ipynb`. 

After running the notebook to execute all of the preprocessing steps, update `data_root_dir` in `./config/vlm/eval/paths/default.yaml` to point to the path where the dataset is saved. Then the dataset can be loaded as follows:
```
dataset = VQARADDataset(
    splits=['train', 'test'], 
    main_split='test',
    hf_cache_dir='/data'
)
```


## 3. Zero-/Few-shot Prompting with Model-Specific Prompt Selection (Finding 1, Section 4)

### 3.1. Medical LLM vs. General-Domain LLM
To evaluate all pairs of medical and general-domain LLMs on all textual QA datasets for zero-shot and 3-shot settings, run the following script:
```
./scripts/eval/llm/compare_medical_general.sh
```
To adjust the settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/llm/eval_few_shot.sh` or `./configs/llm/eval/eval-config.yaml`.

All of the results will be automatically saved under the following directories:
```
./results/llm/<dataset>/<model>/T=0,prompt=zero-shot,constrain_vocab=False,pred_logprob=False,n_seeds=1 # Zero-shot
./results/llm/<dataset>/<model>/T=0,prompt=3-shot,constrain_vocab=False,pred_logprob=False,n_seeds=1 # 3-shot
```
Within each directory, the `test_results.pkl` will contain all of the predictions generated for the test set, along with the exact-match accuracies. The best prompt will be saved as `template_str.yaml` in the jinja2 format.

### 3.2 Medical VLM vs. General-Domain VLM
To evaluate all pairs of medical and general-domain VLMs on all visual QA datasets for zero-shot and 3-shot settings, run the following script:
```
./scripts/eval/vlm/compare_medical_general.sh
```
To adjust the settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/vlm/eval_few_shot.sh` or `./configs/vlm/eval/eval-config.yaml`.

All of the results will be automatically saved under the following directories:
```
./results/vlm/<dataset>/<model>/T=0,prompt=zero-shot,constrain_vocab=False,pred_logprob=False,n_seeds=1 # Zero-shot
./results/vlm/<dataset>/<model>/T=0,prompt=3-shot,constrain_vocab=False,pred_logprob=False,n_seeds=1 # 3-shot
```
Within each directory, the `test_results.pkl` will contain all of the predictions generated for the test set, along with the exact-match accuracies. The best prompt will be saved as `template_str.yaml` in the jinja2 format.

## 4. Zero-/Few-Shot Prompting with Prompt Optimized for the Medical Model Only (Finding 2, Section 4)

### 4.1. Medical LLM vs. General-Domain LLM
After running the experiments in Step 3.1, run the following script:
```
./scripts/eval/llm/compare_medical_general_medopt.sh
```
This script will run all of the zero-shot and 3-shot experiments for the general-domain LLMs using the best prompt formats selected for their medical counterparts from Step 3.1.

To adjust the settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/llm/eval_few_shot_medopt.sh` or `./configs/llm/eval/eval-config.yaml`.

All of the results will be saved in the exact same format as in Step 3.1 and will only update the `test_results.pkl` file with the exact-match accuracies calculated.

### 4.2. Medical VLM vs. General-Domain VLM
After running the experiments in Step 4.1, run the following script:
```
./scripts/eval/vlm/compare_medical_general_medopt.sh
```
This script will run all of the zero-shot and 3-shot experiments for the general-domain VLMs using the best prompt formats selected for their medical counterparts from Step 3.1.

To adjust the settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/vlm/eval_few_shot_medopt.sh` or `./configs/vlm/eval/eval-config.yaml`.

All of the results will be saved in the exact same format as in Step 3.1 and will only update the `test_results.pkl` file with the exact-match accuracies calculated.