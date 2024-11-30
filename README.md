# The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models

<p align="center">
    <img src="./figs/eval-medical-dapt-robots-fighting.webp" alt="image" width="30%">
</p>

<br>n

This is the official repository for the EMNLP 2024 paper (Oral): 
> Daniel P. Jeong, Saurabh Garg, Zachary C. Lipton, and Michael Oberst. [Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress?](https://arxiv.org/abs/2411.04118) *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

and its *extended version* (preprint):
> Daniel P. Jeong, Pranav Mani, Saurabh Garg, Zachary C. Lipton, and Michael Oberst. [The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models](https://arxiv.org/abs/2411.08870). *arXiv:2411.08870*.

*In the extended version, we include additional results on closed-ended QA tasks based on clinical notes in addition to medical-exam-style QA,  as well as a comparison of performance when using medical versus general domain models as an initialization for downstream supervised fine-tuning.*

We include all of the code used for preprocessing the medical QA datasets and running the main zero-/few-shot prompting and supervised fine-tuning experiments discussed in the paper. For details on the overall experimental setup, see Section 3 of the [extended version](https://arxiv.org/abs/2411.08870). For discussion of the results, see Sections 4 (zero-/few-shot prompting) and 5 (supervised fine-tuning) of the [extended version](https://arxiv.org/abs/2411.08870).

## 🔍 Links For Quick Navigation
- 📄 Extended Version of Our Paper: [[arXiv](https://arxiv.org/abs/2411.08870)]
- 📄 EMNLP Version of Our Paper: [[arXiv](https://arxiv.org/abs/2411.04118), [ACL Anthology](https://aclanthology.org/2024.emnlp-main.677/)]
- [🤖 Models](#-models)
    - [LLMs](#llms)
    - [VLMs](#vlms)
- [📁 Datasets](#-datasets)
    - [Textual QA: Medical Knowledge](#textual-qa-medical-knowledge)
    - [Textual QA: Clinical Notes](#textual-qa-clinical-notes)
    - [Visual QA](#visual-qa)
    - [Configuring the Dataset Paths](#configuring-the-dataset-paths)
- [🐍 Setting Up the Conda Environment](#-setting-up-the-conda-environment)
- [📁 Loading the Data](#-loading-the-data)
    - [Textual QA Datasets](#textual-qa-datasets)
    - [Visual QA Datasets](#visual-qa-datasets)
- [📊 Zero-/Few-Shot Prompting Experiments with Model-Specific Prompt Selection (Section 4.1)](#-zero-few-shot-prompting-experiments-with-model-specific-prompt-selection-section-41)
    - [Medical LLM vs. General-Domain LLM](#medical-llm-vs-general-domain-llm)
    - [Medical VLM vs. General-Domain VLM](#medical-vlm-vs-general-domain-vlm)
- [📊 Zero-/Few-Shot Prompting Experiments with Prompt Optimized Only for the Medical Model (Section 4.2)](#-zero-few-shot-prompting-experiments-with-prompt-optimized-only-for-the-medical-model-section-42)
    - [Medical LLM vs. General-Domain LLM](#medical-llm-vs-general-domain-llm-1)
    - [Medical VLM vs. General-Domain VLM](#medical-vlm-vs-general-domain-vlm-1)
- [📊 Supervised Fine-Tuning Experiments (Section 5)](#-supervised-fine-tuning-experiments-section-5)
    - [LoRA Fine-Tuning and Evaluation for LLMs](#lora-fine-tuning-and-evaluation-for-llms)
    - [LoRA Fine-Tuning and Evaluation for LLaVA-Med-7B and LLaVA-v0-7B](#lora-fine-tuning-and-evaluation-for-llava-med-7b-and-llava-v0-7b)
    - [Parameter-Efficient Fine-Tuning and Evaluation for Med-Flamingo-9B and Open-Flamingo-9B](#parameter-efficient-fine-tuning-and-evaluation-for-med-flamingo-9b-and-open-flamingo-9b)
- [🙂 Citing Our Work (BibTeX)](#-citing-our-work-bibtex)

<br>

## 🤖 Models
For all medical and general-domain LLMs and VLMs used for evaluation, we use the HuggingFace checkpoints listed below. For each general-domain LLM/VLM, we list the corresponding medical counterpart(s).

### LLMs
- Llama-3-70B-Instruct: `meta-llama/Meta-Llama-3-70B-Instruct`
    - Med42-v2-70B: `m42-health/Llama3-Med42-70B`
    - OpenBioLLM-70B: `aaditya/Llama3-OpenBioLLM-70B`
- Llama-2-70B: `meta-llama/Llama-2-70b-hf`
    - MediTron-70B: `epfl-llm/meditron-70b`
    - Clinical-Camel-70B: `wanglab/ClinicalCamel-70B`
    - Med42-v1-70B: `m42-health/med42-70b`
- Llama-3-8B-Instruct: `meta-llama/Meta-Llama-3-8B-Instruct`
    - Med42-v2-8B: `m42-health/Llama3-Med42-8B`
- Llama-3-8B: `meta-llama/Meta-Llama-3-8B`
    - OpenBioLLM-8B: `aaditya/Llama3-OpenBioLLM-8B`
- Llama-2-7B: `meta-llama/Llama-2-7b-hf`
    - MediTron-7B: `epfl-llm/meditron-7b`
- Mistral-7B-Instruct-v0.1: `mistralai/Mistral-7B-Instruct-v0.1`
    - BioMistral-7B: `BioMistral/BioMistral-7B`
- Llama-2-7B-Chat: `meta-llama/Llama-2-7b-chat-hf`
    - BioMedGPT-LM-7B: `PharMolix/BioMedGPT-LM-7B`

### VLMs
- LLaVA-v0-7B: `liuhaotian/LLaVA-7b-delta-v0`
    - LLaVA-Med-7B: `microsoft/llava-med-7b-delta`
- Open-Flamingo-9B: `openflamingo/OpenFlamingo-9B-deprecated`
    - Med-Flamingo-9B: `med-flamingo/med-flamingo`

For LLaVA-Med-7B and LLaVA-v0-7B, we note that the checkpoints provided are delta weights that cannot be used directly. Please see the instructions provided in the [LLaVA-Med repository](https://github.com/microsoft/LLaVA-Med/tree/b9a98a736d2ef05bcf5ff345be6403fb3a664eaf?tab=readme-ov-file#training) and the [LLaVA repository](https://github.com/haotian-liu/LLaVA) for merging the delta weights with the base LLaMA-7B LLM weights: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#legacy-models-delta-weights.

<br>

## 📁 Datasets
As detailed in Section 3 and Appendix A.1 of the [extended version](https://arxiv.org/abs/2411.08870), we use the following datasets for evaluation, listed below. For details on how each dataset was preprocessed, see Section 3 and Appendix A.1 in the paper. 

### Textual QA: Medical Knowledge
- MedQA (4 Options & 5 Options) (Jin et al., 2020) [[HuggingFace](https://huggingface.co/datasets/bigbio/med_qa)]
- MedMCQA (Pal et al., 2022) [[HuggingFace](https://huggingface.co/datasets/openlifescienceai/medmcqa)]
- PubMedQA (Jin et al., 2019) [[HuggingFace](https://huggingface.co/datasets/bigbio/pubmed_qa)]
- MMLU-Medical (Hendrycks et al., 2021) [[HuggingFace](https://huggingface.co/datasets/lukaemon/mmlu)]
    - Subset of MMLU corresponding to 9 subjects related to medicine: anatomy, clinical knowledge, college biology, college medicine, high school biology, medical genetics, nutrition, professional medicine, virology

All of the textual medical knowledge QA datasets are directly accessible via HuggingFace (links included above).

### Textual QA: Clinical Notes
- MedNLI (Romanov and Shivade, 2018) [[PhysioNet](https://physionet.org/content/mednli/1.0.0/)]
- EHRNoteQA (Kweon et al., 2024) [[PhysioNet](https://physionet.org/content/ehr-notes-qa-llms/1.0.1/)]
- 2008 i2b2 Obesity Comorbidity Detection Challenge (Uzuner, 2009)
    - 4 binary classification tasks: asthma, CAD, diabetes, obesity
- CASI Clinical Acronym Sense Disambiguation (Moon et al., 2014) [[Official Website](https://conservancy.umn.edu/items/6651323b-444a-479e-a41a-abca58c2e721)]
- MIMIC-III Clinical Acronym Sense Disambiguation (Johnson et al., 2016; Adams et al., 2020) [[PhysioNet](https://physionet.org/content/mimiciii/1.4/)]

Except for the CASI dataset, all of the textual clinical note QA datasets require credentialed access. The i2b2 dataset can be accessed via the [Harvard DBMI Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). The remaining datasets are all available via PhysioNet (links included above). Note that EHRNoteQA also requires downloading the clinical notes in MIMIC-IV (Johnson et al., 2020; [PhysioNet](https://physionet.org/content/mimic-iv-note/2.2/)). To gain credentials for PhysioNet, follow the instructions [here](https://mimic.mit.edu/docs/gettingstarted/#physionet-credentialing).

**Additional Preprocessing Steps:**
- MedNLI: No additional preprocessing steps are required after downloading from PhysioNet.
- EHRNoteQA: After downloading the data (also the MIMIC-IV clinical notes), follow the instructions in the [official repository](https://github.com/ji-youn-kim/EHRNoteQA).
- i2b2 classification datasets: After downloading the raw data, follow the steps described in `./notebooks/preprocess-i2b2.ipynb`, which was adapted from Ceballos-Arroyo et al. (2024).
- CASI: After downloading the raw data, run the preprocessing pipeline from Adams et al. (2020) in their [GitHub repository](https://github.com/griff4692/LMC). Then follow the steps described in `./notebooks/preprocess-mimic-iii.ipynb`.
- MIMIC-III: After downloading the MIMIC-III clinical notes, run the preprocessing pipeline from Adams et al. (2020) in their [GitHub repository](https://github.com/griff4692/LMC). Then follow the steps described in `./notebooks/preprocess-mimic-iii.ipynb`.

### Visual QA
- VQA-RAD (Lau et al., 2018) [[Official Website](https://osf.io/89kps/)]
- PathVQA (He et al., 2020) [[GitHub Repository](https://github.com/UCSD-AI4H/PathVQA)]
- SLAKE (Liu et al., 2021) [[Official Website](https://www.med-vqa.com/slake/)]
- MMMU-Medical (Yue et al., 2024) [[HuggingFace](https://huggingface.co/datasets/MMMU/MMMU)]
    - Subset of MMMU corresponding to 5 subjects relevant to medicine: basic medical science, clinical medicine, diagnostics & laboratory medicine, pharmacy, public health

All of the visual QA datasets are publicly available (links included above). 

**Additional Preprocessing Steps:** For VQA-RAD, PathVQA, and SLAKE, follow the steps in `./notebooks/preprocess-vqa.ipynb`.

### Configuring the Dataset Paths
For all datasets, make sure to appropriately update the dataset config files (e.g., `./configs/llm/eval/dataset/mednli.yaml`) and the default path config files (e.g., `./configs/llm/eval/paths/default.yaml`) to point to the correct paths where you have downloaded the data.

<br>

## 🐍 Setting Up the Conda Environment
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

<br>

## 📁 Loading the Data

### Textual QA Datasets
For all of the textual QA datasets that are available via HuggingFace, instantiating the relevant dataset class (e.g., `MedQADataset`, `MMLUMedicalDataset`) in `./src/dataset.py` will automatically download and cache the data to the path specified by the `hf_cache_dir` argument:
```python
dataset = MedQADataset(
    name='medqa', # 5 options (for 4 options, use `medqa-usmle`)
    splits=['train', 'test'], 
    main_split='test',
    hf_cache_dir='/data'
)
```
You can also load the other datasets that require manual downloading and preprocessing beforehand in the same way, but be sure to update the paths in the dataset config files appropriately.

For zero-shot prompting, running the following will apply a prompt template specified in the argument to all of the QA examples in the `main_split` in the zero-shot format (i.e., system prompt + question):
```python
dataset.load_and_apply_prompt_template(
    model_name='llama-3-8b', # Use the default prompt format for Llama-3-8B
    prompt_type='zero-shot', # Zero-shot prompting format
    tokenizer=tokenizer # Assuming model tokenizer has been loaded beforehand
)
```
To randomly sample a prompt format using the context-free grammar we discuss in Section 3 and Appendix B, you can additionally pass the `sample_kwargs` argument to the dataset class, with the desired fixed random seeds.
```python
dataset.load_and_apply_prompt_template(
    model_name='llama-3-8b', # Use the default prompt format for Llama-3-8B
    sample_kwargs=dict(prompt_template_seed=0)
    prompt_type='zero-shot', 
    tokenizer=tokenizer 
)
```
For few-shot prompting, call the `sample_few_shot_qas()` method before calling `load_and_apply_prompt_template()`:
```python
dataset = MedQADataset(
    splits=['train', 'test'], 
    main_split='test',
    hf_cache_dir='/data'
)
dataset.sample_few_shot_qas(n_shot=3, seed=0)
dataset.load_and_apply_prompt_template(
    model_name='llama-3-8b',
    sample_kwargs=dict(prompt_template_seed=0)
    prompt_type='few-shot', 
    tokenizer=tokenizer 
)
```

### Visual QA Datasets
The MMMU-Medical datasets can be directly loaded from HuggingFace, as with all of the textual medical QA datasets. Below is an example for loading the MMMU (Basic Medical Science) dataset for 3-shot prompting LLaVA-Med-7B:
```python
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

All other visual QA datasets should be downloaded separately from the official repositories beforehand, as detailed [here](#visual-qa). For these datasets, which contain both closed-ended and open-ended QA examples, we performed additional preprocessing to only extract the closed-ended QA examples and format them into structured `.jsonl` files, as detailed in `./notebooks/preprocess-vqa.ipynb`. 

After running the notebook to execute all of the preprocessing steps, update `data_root_dir` in `./config/vlm/eval/paths/default.yaml` to point to the path where the dataset is saved. Then the dataset can be loaded as follows (showing the VQA-RAD dataset as an example):
```python
dataset = VQARADDataset(
    splits=['train', 'test'], 
    main_split='test',
    hf_cache_dir='/data'
)
```

<br>

## 📊 Zero-/Few-Shot Prompting Experiments with Model-Specific Prompt Selection (Section 4.1)

### Medical LLM vs. General-Domain LLM
To evaluate all pairs of medical and general-domain LLMs on all textual QA datasets in the zero-shot and 3-shot settings, run the following script:
```bash
./scripts/eval/llm/compare_medical_general.sh "<gpu_indices>" "<decoding>" "<prompt_optimization_flag>"
```
- To run the experiments with greedy decoding, set the `<decoding>` argument to "greedy". For constrained decoding, set it to "logprob". 
- To optimize the prompt for each model, set the `<prompt_optimization_flag>` argument to "true". If set to "false", evaluations will be done based on the default prompts.
- To adjust other settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/llm/eval_few_shot.sh` or `./configs/llm/eval/eval-config.yaml`.

All of the results will be automatically saved under the following directories (the brackets are placeholders):
```bash
# Greedy decoding
./results/llm/<dataset>/<model>/T=0,prompt=zero-shot,constrain_vocab=False,n_seeds=1 # Zero-shot
./results/llm/<dataset>/<model>/T=0,prompt=3-shot,constrain_vocab=False,n_seeds=1 # 3-shot

# Constrained decoding
./results/llm/<dataset>/<model>/T=0,prompt=zero-shot,constrain_vocab=True,n_seeds=1 # Zero-shot
./results/llm/<dataset>/<model>/T=0,prompt=3-shot,constrain_vocab=True,n_seeds=1 # 3-shot
```
Within each directory, the `test_results.pkl` will contain all of the predictions generated for the test set, along with the exact-match accuracies. The best prompt will be saved as `template_str.yaml` in the jinja2 format.

### Medical VLM vs. General-Domain VLM
To evaluate all pairs of medical and general-domain VLMs on all visual QA datasets in the zero-shot and 3-shot settings, run the following script:
```bash
./scripts/eval/vlm/compare_medical_general.sh "<gpu_indices>" "<decoding>" "<prompt_optimization_flag>"
```
- To run the experiments with greedy decoding, set the `<decoding>` argument to "greedy". For constrained decoding, set it to "logprob". 
- To optimize the prompt for each model, set the `<prompt_optimization_flag>` argument to "true". If set to "false", evaluations will be done based on the default prompts.
- To adjust other settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/vlm/eval_few_shot.sh` or `./configs/vlm/eval/eval-config.yaml`.

All of the results will be automatically saved under the following directories:
```bash
# Greedy decoding
./results/vlm/<dataset>/<model>/T=0,prompt=zero-shot,constrain_vocab=False,n_seeds=1 # Zero-shot
./results/vlm/<dataset>/<model>/T=0,prompt=3-shot,constrain_vocab=False,n_seeds=1 # 3-shot

# Constrained decoding
./results/vlm/<dataset>/<model>/T=0,prompt=zero-shot,constrain_vocab=True,n_seeds=1 # Zero-shot
./results/vlm/<dataset>/<model>/T=0,prompt=3-shot,constrain_vocab=True,n_seeds=1 # 3-shot
```
Within each directory, the `test_results.pkl` will contain all of the predictions generated for the test set, along with the exact-match accuracies. The best prompt will be saved as `template_str.yaml` in the jinja2 format.

<br>

## 📊 Zero-/Few-Shot Prompting Experiments with Prompt Optimized Only for the Medical Model (Section 4.2)

### Medical LLM vs. General-Domain LLM
After running the LLM experiments with independent prompt selection, run the following script:
```bash
./scripts/eval/llm/compare_medical_general_medopt.sh "<gpu_indices>" "<decoding>"
```
- This script will run all of the zero-shot and 3-shot experiments for the general-domain LLMs using the best prompt formats selected for each of their medical counterparts.
- To run the experiments with greedy decoding, set the `<decoding>` argument to "greedy". For constrained decoding, set it to "logprob". 
- To adjust other settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/llm/eval_few_shot_medopt.sh` or `./configs/llm/eval/eval-config.yaml`.

All of the results will be saved in the exact same format as before and will only update the `test_results.pkl` file with the exact-match accuracies calculated. In the .pkl file, the corresponding entries will have the additional `_med` suffix to distinguish them from the results of the independent prompt selection experiments.

### Medical VLM vs. General-Domain VLM
After running the VLM experiments with independent prompt selection, run the following script:
```bash
./scripts/eval/vlm/compare_medical_general_medopt.sh "<gpu_indices>" "<decoding>"
```
- This script will run all of the zero-shot and 3-shot experiments for the general-domain VLMs using the best prompt formats selected for their medical counterparts.
- To run the experiments with greedy decoding, set the `<decoding>` argument to "greedy". For constrained decoding, set it to "logprob". 
- To adjust other settings for loading and running the models (e.g., number of GPUs, proportion of total GPU memory to reserve), either modify the sampling configurations specified in `./scripts/eval/vlm/eval_few_shot_medopt.sh` or `./configs/vlm/eval/eval-config.yaml`.

All of the results will be saved in the exact same format as before and will only update the `test_results.pkl` file with the exact-match accuracies calculated. In the .pkl file, the corresponding entries will have the additional `_med` suffix to distinguish them from the results of the independent prompt selection experiments.

<br>

## 📊 Supervised Fine-Tuning Experiments (Section 5)

### LoRA Fine-Tuning and Evaluation for LLMs
To fine-tune a given medical/general-domain LLM on a textual QA dataset, run the following script:
```bash
./scripts/hpo/llm/run_lora_hpo.sh "<model>" "<dataset>" "<lora_r>" "<n_nodes>" "<head_node_ip>" "<gpu_indices>"
```
- The script is designed to run a sweep (over the learning rates, as discussed in Section 3.2 and Appendix D) for each LoRA rank (`<lora_r>`).
- To adjust the search space for the learning rates or the other hyperparameters, modify the command-line arguments specified inside the script.
- For each hyperparameter trial, the checkpoint will by default be saved under the `./ckpts/llm/` folder, which stores the model weights that achieved the lowest validation loss during training.

After running all of the sweeps, run the following script to select the *best* checkpoint across all hyperparameter trials:
```bash
./scripts/eval/llm/find_best_model.sh "<model>" "<dataset>" "<ft_method>" "<n_gpus>"
```
- For the LLMs, set the `<ft_method>` argument to "lora".
- The `<n_gpus>` argument should correspond to the number of GPUs used for training, and is mainly used to recover the batch size used during training.
- Running this script will automatically create a config file for the best checkpoint under `./configs/llm/eval/model` with the following name: `<model>_lora-<dataset>-best.yaml`.
- There is also the option to remove all of the checkpoints except for the best one by adding the `--clear_ckpts` flag inside the `find_best_model.sh` script.

To run the final evaluation with the best checkpoint, run the following script:
```bash
./scripts/eval/llm/eval_finetuned.sh "<model>" "<dataset>" "<ft_method>" "<n_gpus>"
```
The evaluation result will be saved in the `test_results.pkl` file under the `./results/llm/<dataset>/<model>_lora-<dataset>-best/T=0,prompt=zero-shot,constrain_vocab=False,n_seeds=1` directory.

### LoRA Fine-Tuning and Evaluation for LLaVA-Med-7B and LLaVA-v0-7B
To fine-tune LLaVA-Med-7B or LLaVA-v0-7B on a visual QA dataset, run the following script:
```bash
./scripts/hpo/vlm/run_lora_hpo.sh "<model>" "<dataset>" "<lora_r>" "<n_nodes>" "<head_node_ip>" "<gpu_indices>"
```
- The script is designed to run a sweep (over the learning rates, as discussed in Section 3.2 and Appendix D) for each LoRA rank (`<lora_r>`).
- To adjust the search space for the learning rates or the other hyperparameters, modify the command-line arguments specified inside the script.
- For each hyperparameter trial, the checkpoint will by default be saved under the `./ckpts/vlm/` folder, which stores the model weights that achieved the lowest validation loss during training.
- Note that LoRA fine-tuning is not implemented for Med-Flamingo-9B and Open-Flamingo-9B in the current implementation.

After running all of the sweeps, run the following script to select the *best* checkpoint across all hyperparameter trials: 
```bash
./scripts/eval/vlm/find_best_model.sh "<model>" "<dataset>" "<ft_method>" "<n_gpus>"
```
- For LLaVA-Med-7B and LLaVA-v0-7B, set the `<ft_method>` argument to "lora".
- The `<n_gpus>` argument should correspond to the number of GPUs used for training, and is mainly used to recover the batch size used during training.
- Running this script will automatically create a config file for the best checkpoint under `./configs/vlm/eval/model` with the following name: `<model>_lora-<dataset>-best.yaml`.
- There is also the option to remove all of the checkpoints except for the best one by adding the `--clear_ckpts` flag inside the `find_best_model.sh` script.

To run the final evaluation with the best checkpoint, run the following script:
```bash
./scripts/eval/vlm/eval_finetuned.sh "<model>" "<dataset>" "<ft_method>" "<n_gpus>"
```
The evaluation result will be saved in the `test_results.pkl` file under the `./results/vlm/<dataset>/<model>_lora-<dataset>-best/T=0,prompt=zero-shot,constrain_vocab=False,n_seeds=1` directory.

### Parameter-Efficient Fine-Tuning and Evaluation for Med-Flamingo-9B and Open-Flamingo-9B
To fine-tune Med-Flamingo-9B or Open-Flamingo-9B on a visual QA dataset, run the following script:
```bash
./scripts/hpo/vlm/run_ft_hpo.sh "<model>" "<dataset>" "<gpu_indices>"
```
- The script is designed to run the full sweep (over both the learning rates and weight decay coefficients, as discussed in Section 3.2 and Appendix D).
- To adjust the search space for the learning rates or the weight decay coefficients, modify the command-line arguments specified inside the script.
- For each hyperparameter trial, the checkpoint will by default be saved under the `./ckpts/vlm/` folder, which stores the model weights that achieved the lowest validation loss during training.

After running all of the sweeps, run the following script to select the *best* checkpoint across all hyperparameter trials:
```bash
./scripts/eval/vlm/find_best_model.sh "<model>" "<dataset>" "<ft_method>" "<n_gpus>"
```
- For Med-Flamingo-9B and Open-Flamingo-9B, set the `<ft_method>` argument to "ft".
- The `<n_gpus>` argument should correspond to the number of GPUs used for training, and is mainly used to recover the batch size used during training.
- Running this script will automatically create a config file for the best checkpoint under `./configs/vlm/eval/model` with the following name: `<model>_ft-<dataset>-best.yaml`.
- There is also the option to remove all of the checkpoints except for the best one by adding the `--clear_ckpts` flag inside the `find_best_model.sh` script.

To run the final evaluation with the best checkpoint, run the following script:
```bash
./scripts/eval/vlm/eval_finetuned.sh "<model>" "<dataset>" "<ft_method>" "<n_gpus>"
```
The evaluation result will be saved in the `test_results.pkl` file under the `./results/vlm/<dataset>/<model>_ft-<dataset>-best/T=0,prompt=zero-shot,constrain_vocab=False,n_seeds=1` directory.

<br>

## 🙂 Citing Our Work (BibTeX)
```bibtex
# EMNLP 2024 Version
@inproceedings{jeong-etal-2024-medical,
    title = "Medical Adaptation of Large Language and Vision-Language Models: Are We Making Progress?",
    author = "Jeong, Daniel P and Garg, Saurabh and Lipton, Zachary Chase and Oberst, Michael",
    editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.677",
    doi = "10.18653/v1/2024.emnlp-main.677",
    pages = "12143--12170"
}

# Extended Version
@article{jeong-etal-2024-limited,
    title = "The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models",
    author = "Jeong, Daniel P and Mani, Pranav and Garg, Saurabh and Lipton, Zachary Chase and Oberst, Michael",
    journal = "arXiv preprint arXiv:2411.08870",
    year = "2024"
}
```
