import os
import os.path as osp
import pickle
import dill
import json
import argparse
import numpy as np
import re
import yaml
from copy import deepcopy
import pandas as pd
from collections import Counter
from functools import partial
from datetime import datetime
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

import huggingface_hub

import torch
import transformers
from transformers import StoppingCriteria
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from accelerate import Accelerator
from accelerate.utils import gather_object

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from peft import PeftModel

from omegaconf import OmegaConf

import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset import (
    MedNLIDataset,
    EHRNoteQADataset,
    n2c2Dataset,
    CASISenseDataset,
    MIMICIIISenseDataset,
    MedQADataset,
    MedMCQADataset,
    PubMedQADataset,
    MMLUMedicalDataset
)

from utils import LLAMA2_CHAT_TEMPLATE, LLAMA3_CHAT_TEMPLATE, MISTRAL_CHAT_TEMPLATE
from utils import get_llm_system_prompt, sample_llm_prompt_template
from utils import remove_punc, white_space_fix, normalize_text

# Default paths
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
EVAL_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/llm/eval')

DEFAULT_SEED = 42
MODELS_REQ_LOGIN = [
    'meta-llama/Meta-Llama-3-70B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-70b-hf',
    'epfl-llm/meditron-7b',
    'epfl-llm/meditron-70b',
    'mistralai/Mistral-7B-Instruct-v0.1',
    'wanglab/ClinicalCamel-70B',
    'm42-health/med42-70b',
    'm42-health/Llama3-Med42-70B'
    'm42-health/Llama3-Med42-8B',
]
DATASET_NAME_TO_CLS = {
    'mednli': MedNLIDataset,
    'ehrnoteqa': EHRNoteQADataset,
    'n2c2_2008-obesity_asthma': n2c2Dataset,
    'n2c2_2008-obesity_cad': n2c2Dataset,
    'n2c2_2008-obesity_diabetes': n2c2Dataset,
    'n2c2_2008-obesity_obesity': n2c2Dataset,
    'casi-sense': CASISenseDataset,
    'mimic-iii-sense': MIMICIIISenseDataset,
    'medqa': MedQADataset,
    'medqa-usmle': MedQADataset,
    'medmcqa': MedMCQADataset,
    'pubmedqa': PubMedQADataset,
    'mmlu_anatomy': MMLUMedicalDataset,
    'mmlu_clinical-knowledge': MMLUMedicalDataset,
    'mmlu_college-biology': MMLUMedicalDataset,
    'mmlu_college-medicine': MMLUMedicalDataset,
    'mmlu_medical-genetics': MMLUMedicalDataset,
    'mmlu_professional-medicine': MMLUMedicalDataset,
    'mmlu_high-school-biology': MMLUMedicalDataset,
    'mmlu_virology': MMLUMedicalDataset,
    'mmlu_nutrition': MMLUMedicalDataset
}
COMPONENTS_TO_FIX_FOR_SAMPLING = {
    'biomistral-7b': ['system_prompt', 'prompt_template'],
    
    'meditron-70b': ['system_prompt'], # NOTE: Template provided by authors did not work very well
    'meditron-7b': ['system_prompt'], # NOTE: Template provided by authors did not work very well
    'llama-3-70b-instruct': ['system_prompt'],
    'llama-3-8b-instruct': ['system_prompt'],
    'llama-3-8b': ['system_prompt'],
    'openbiollm-70b': ['system_prompt'],
    'openbiollm-8b': ['system_prompt'],
    'med42-v2-70b': ['system_prompt'],
    'med42-v2-8b': ['system_prompt'],
    'med42-v1-70b': ['system_prompt', 'prompt_template'],
    'mistral-7b-v0.1': ['system_prompt'],
    'clinical-camel-70b': ['system_prompt'],
    'llama-2-70b': ['system_prompt'],
    'llama-2-7b': ['system_prompt'],
    'llama-2-7b-chat': ['system_prompt'],
    'biomedgpt-7b': ['system_prompt'],
}
GENERAL_MODEL_TO_MEDICAL_MODEL = {
    'llama-3-70b-instruct': ['openbiollm-70b', 'med42-v2-70b'],
    'llama-2-70b': ['meditron-70b', 'clinical-camel-70b', 'med42-v1-70b'],
    'llama-3-8b-instruct': 'med42-v2-8b',
    'llama-3-8b': 'openbiollm-8b',
    'mistral-7b-v0.1': 'biomistral-7b',
    'llama-2-7b': 'meditron-7b',
    'llama-2-7b-chat': 'biomedgpt-7b'
}

'''
    Exact-match accuracy evaluation functions.

'''

def extract_pred(output, answer, options):
    '''Extracts the prediction from the output text.'''
    
    option_letters = [chr(ord('A') + i) for i in range(len(options))]
    answer_idx = options.index(answer)
    answer_letter = chr(ord('A') + answer_idx)
    pred = None

    # Check if the letter was predicted
    # e.g., r'(?:^|\s|\(|\[|<)(A|B|C|D)(?=\s|$|\.|<\/s>|\)|\]|>)'
    start_chr = 'A'
    pattern = '(' + '|'.join([chr(ord(start_chr)+i) for i, _ in enumerate(options)]) + ')' #+ '|'
    pattern = r'(?:^|\s|\(|\[|<){}(?=\s|$|\.|<\/s>|\)|\]|>)'.format(pattern)
    pattern = re.compile(pattern)
    candidates = pattern.findall(output)

    # If there is a single letter prediction made
    if len(candidates) == 1:
        pred = candidates[0]

    # Handle cases where the model repeats the answer choices
    elif len(candidates) > 1:
        parsed_candidates = []
        for candidate in candidates:
            for option_letter in option_letters:
                if option_letter in candidate:
                    parsed_candidates.append(option_letter)

        if len(set(parsed_candidates)):
            pred = parsed_candidates[0]
        else:
            pred = None

    # Check if the fully expanded answer is in the output
    # e.g., if the correct option was "(A) heart rate", we check for "heart rate"
    else:
        # Remove punctuations, make lowercase, and reduce multiple whitespaces to single space
        norm_answer = white_space_fix(remove_punc(answer).strip().lower())
        norm_output = white_space_fix(remove_punc(output).strip().lower())

        pattern = r'(?:^|\s){}(?=\s|$|\.|<\/s>)'.format(re.escape(norm_answer))
        pattern = re.compile(pattern)
        candidates = pattern.findall(norm_output)

        # If there is a match, check that other answer choices are not generated
        if len(candidates) > 0:
            if norm_answer in candidates[0]:
                pred = answer_letter

            for norm_option in [white_space_fix(remove_punc(o).strip().lower()) for o in options if o != answer]:
                pattern = r'(?:^|\s){}(?=\s|$|\.|<\/s>)'.format(re.escape(norm_option))
                pattern = re.compile(pattern)
                candidates = pattern.findall(norm_output)

                if len(candidates) > 0:
                    pred = None
                    break

    return pred

def evaluate_accuracy_exact_match(results, verbose=False):
    '''Evaluates the accuracy by exactly matching the generated token(s) and the ground-truth answer.'''

    n_qa_pairs = len(results['questions'])
    evals = []

    if verbose:
        logging.info('Evaluating exact-match accuracy...')

    for i in range(n_qa_pairs):
        options = results['options'][i]
        answer = results['answers'][i]
        answer_idx = options.index(answer)
        answer_letter = chr(ord('A') + answer_idx)
        
        # Fetch predictions and take majority vote
        preds = [
            extract_pred(output.strip(), answer, options)
            for output in results['outputs'][i]
        ]
        preds = [p for p in preds if p is not None]
        
        if len(preds) == 0: # No valid prediction found
            evals.append(0)
        else:
            counts = Counter(preds)
            max_count = max(counts.values())
            max_preds = [k for k,v in counts.items() if v == max_count]
            max_preds.sort(key=lambda x: preds.index(x))
            majority_pred = max_preds[0]
            evals.append(int(majority_pred == answer_letter))

    evals = np.array(evals)
    accuracies = np.mean(evals)

    return evals, accuracies

'''
    Keyword stopping criteria.

'''

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0

        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            
            # Skip bos_token_id
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1] # Length of input sequence

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # The number of newly generated tokens, capped at the maximum keyword length
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0,-keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
            
        outputs = self.tokenizer.batch_decode(output_ids[:,-offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
            
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []

        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))

        return all(outputs)

'''
    Main inference and evaluation functions. By default, we use the vLLM approach.

'''

def main_accelerate(args):
    '''Main LLM inference function based on accelerate.'''

    start_time = datetime.now()
    
    accelerator = Accelerator()
    device = accelerator.device

    # Quantization config
    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type
        )
    else:
        quantization_config = None
    
    # Load model and tokenizer
    if 'lora' in args.model.name:
        adapter_config_path = osp.join(args.model.path, 'adapter_config.json')
        adapter_config = json.load(open(adapter_config_path))
        base_model_path = adapter_config['base_model_name_or_path']

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            cache_dir=args.paths.hf_cache_dir,
            quantization_config=quantization_config,
            attn_implementation=args.attn_implementation,
            torch_dtype=(torch.bfloat16 if args.bf16 else None),
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.path,
            cache_dir=args.paths.hf_cache_dir,
            use_fast=True
        )

        # Check if there is a mismatch between tokenizer vocab size and embedding matrix
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            logging.warning('Mismatch between vocab size and embedding matrix.')
            model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, args.model.path)
        logging.info('LoRA weights loaded.')
    else:
        base_model_path = args.model.path
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            cache_dir=args.paths.hf_cache_dir,
            quantization_config=quantization_config,
            attn_implementation=args.attn_implementation,
            torch_dtype=(torch.bfloat16 if args.bf16 else None),
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=args.paths.hf_cache_dir,
            use_fast=True
        )

    tokenizer.padding_side = 'left'

    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if getattr(tokenizer, 'model_max_length', None) is None or tokenizer.model_max_length == VERY_LARGE_INTEGER:
        tokenizer.model_max_length = model.config.max_position_embeddings

    # Adjust context window config for Med42-v1-70B
    # NOTE: Model card mentions that context window size is 4k, but the config file 
    if 'med42-70b' in base_model_path.lower() and 'llama3' not in base_model_path.lower():
        tokenizer.model_max_length = 4096

    # ClinicalCamel: Add Llama-2 tokenizer chat template
    if 'camel' in base_model_path.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

    # OpenBioLLM: Add Llama-3 tokenizer chat template
    elif 'openbiollm' in base_model_path.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    # Mistral: Adjust chat template
    elif 'mistral-7b' in base_model_path.lower() and 'biomistral' not in base_model_path.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    
    model.eval()
    accelerator.wait_for_everyone()

    def _infer(qas, raw_qas, add_stop_words=None):
        questions, options, answers = zip(*raw_qas)
        output_texts = [] # Generated texts
        confidences = [] # Full softmax scores over options
        pred_confidences = [] # Confidence scores for the predicted option

        with accelerator.split_between_processes(list(zip(range(len(qas)), qas))) as acc_qas:
            pbar = tqdm(acc_qas, disable=(not accelerator.is_main_process), total=len(acc_qas))
            for idx, prompt in pbar:
                tokenized = tokenizer(prompt, return_tensors='pt')
                input_ids = tokenized.input_ids.to(device)
                attn_masks = tokenized.attention_mask.to(device)
                
                # NOTE: eos_token_id is excluded from bad_words_ids if present, when generating tokens
                # Reference: https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/generation/logits_process.py#L1253
                # Reference: https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/generation/logits_process.py#L1044
                # Passing bad_words_ids = adding -inf bias to the logits corresponding to the specified token IDs
                if args.constrain_vocab:
                    whitelist = [tokenizer.convert_tokens_to_ids(chr(ord('A')+i)) for i in range(len(options[idx]))]
                    bad_words_ids = [[i] for i in range(len(tokenizer)) if i not in whitelist]
                else:
                    bad_words_ids = None

                seeds = [DEFAULT_SEED] if args.temperature == 0 else np.arange(args.n_seeds)
                do_sample = not (args.temperature == 0)
                
                qa_output_texts = []
                qa_confidences = []
                qa_pred_confidences = []
                for seed in seeds:
                    transformers.set_seed(seed)

                    stop_words = [tokenizer.decode(tokenizer.eos_token_id), '###']
                    if any([m in args.model.name for m in ['llama-3', 'openbiollm', 'med42-v2']]):
                        stop_words.append('<|eot_id|>')

                    if add_stop_words is not None:
                        stop_words += add_stop_words

                    stop_criteria = KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)

                    # Generate output tokens
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids,
                            attention_mask=attn_masks,
                            do_sample=do_sample,
                            temperature=(args.temperature if do_sample else None),
                            top_p=(args.top_p if do_sample else None),
                            num_beams=args.num_beams,
                            max_new_tokens=(1 if args.constrain_vocab else args.max_new_tokens),
                            use_cache=True,
                            bad_words_ids=bad_words_ids,
                            output_scores=True,
                            return_dict_in_generate=True,
                            pad_token_id=tokenizer.pad_token_id,
                            stopping_criteria=[stop_criteria]
                        )
                
                    # Decode output tokens
                    input_token_len = input_ids.shape[1]
                    output_ids = outputs.sequences[:,input_token_len:]
                    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    output_text = output_text.strip()

                    for stop_word in stop_words:
                        if output_text.endswith(stop_word):
                            output_text = output_text[:-len(stop_word)]
                            break
                
                    # Constrain the output vocabulary
                    if args.constrain_vocab:
                        logits = outputs.scores[0].squeeze(0).clone()
                        eos_bias = torch.zeros_like(logits)
                        eos_bias[tokenizer.eos_token_id] = -torch.inf
                        logits += eos_bias
                        probs = torch.softmax(logits, dim=0)
                        option_idxs = [tokenizer.convert_tokens_to_ids(chr(ord('A')+i)) for i in range(len(options[idx]))]
                        option_probs = probs[option_idxs]
                        Z = torch.sum(option_probs)
                        option_probs /= Z # Normalize

                        # Take the argmax of token probabilities
                        if args.temperature == 0:
                            output_idx = torch.argmax(option_probs)
                            output_text = chr(ord('A') + output_idx)
                            confidence = torch.max(option_probs).item()
                        
                        # Sample from the constrained vocabulary
                        elif args.temperature > 0:
                            torch.manual_seed(seed)
                            output_idx = torch.multinomial(option_probs, 1).item()
                            output_text = chr(ord('A') + output_idx)
                            confidence = option_probs[output_idx].item()

                        qa_confidences.append(option_probs.cpu().numpy())
                        qa_pred_confidences.append(confidence)
  
                    qa_output_texts.append(output_text)

                output_texts.append(qa_output_texts)
                confidences.append(qa_confidences)
                pred_confidences.append(qa_pred_confidences)

        results = dict(
            questions=questions, 
            outputs=gather_object(output_texts), 
            answers=answers, 
            options=options, 
            confidences=gather_object(confidences),
            pred_confidences=gather_object(pred_confidences)
        )

        return results

    # Default dataset arguments
    dataset_cls = DATASET_NAME_TO_CLS[args.dataset.name]
    dataset_args = dict(
        name=args.dataset.name,
        qa_dir=args.dataset.path,
        hf_cache_dir=args.paths.hf_cache_dir
    )

    # Name of the model to use for fetching the prompt template for final evaluation
    prompt_model_name = args.model.name

    # Optimize the prompt based on validation performance
    if args.optimize_prompt:
        logging.info('Running prompt optimization...')
        splits = ['train', 'val'] if args.prompt_type == 'few-shot' else ['val']
        main_split = 'val'

        # Load validation dataset
        val_dataset = dataset_cls(splits=splits, main_split=main_split, **dataset_args)

        # Keep track of best template
        val_acc_dict = {
            'system_prompt_seed': [], 
            'prompt_template_seed': [], 
            'few_shot_seed': [], 
            'acc': []
        }
        best_system_prompt = None
        best_template = None
        best_few_shot_seed = None
        best_val_acc = -np.inf

        # Randomly sample prompts
        if args.model.name in COMPONENTS_TO_FIX_FOR_SAMPLING.keys():
            fix_system_prompt = 'system_prompt' in COMPONENTS_TO_FIX_FOR_SAMPLING[args.model.name]
            system_prompt_seeds = [None] if fix_system_prompt else [None] + list(range(args.n_system_prompt_seeds))
            
            fix_prompt_template = 'prompt_template' in COMPONENTS_TO_FIX_FOR_SAMPLING[args.model.name]
            prompt_template_seeds = [None] if fix_prompt_template else [None] + list(range(args.n_prompt_template_seeds))
        else:
            system_prompt_seeds = [None] + list(range(args.n_system_prompt_seeds))
            prompt_template_seeds = [None] + list(range(args.n_prompt_template_seeds))

        few_shot_seeds = list(range(args.n_few_shot_seeds)) if args.prompt_type == 'few-shot' else [DEFAULT_SEED]

        sample_kwargs_list = [
            dict(system_prompt_seed=s1, prompt_template_seed=s2, few_shot_seed=s3)
            for s1 in system_prompt_seeds
            for s2 in prompt_template_seeds
            for s3 in few_shot_seeds
        ]

        for sample_kwargs in sample_kwargs_list:
            system_prompt_seed = sample_kwargs['system_prompt_seed']
            template_seed = sample_kwargs['prompt_template_seed']
            few_shot_seed = sample_kwargs['few_shot_seed']
            
            if args.prompt_type == 'few-shot':
                val_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed)
            
            val_dataset.load_and_apply_prompt_template(
                model_name=args.model.name,
                prompt_type=args.prompt_type,
                tokenize=False,
                tokenizer=tokenizer,
                sample_kwargs=sample_kwargs
            )
            val_qas = val_dataset.qas['val']
            val_qa_dict = val_dataset.qa_dict['val']

            # Add question, options, and answer headers to stop word list to avoid repetitive completions in the output
            prompt_template_str = val_dataset.prompt_template['template_str']
            q_header = prompt_template_str['q_header']
            o_header = prompt_template_str['o_header']
            a_header = prompt_template_str['a_header']
            
            headers = []
            headers += q_header if isinstance(q_header, list) else [q_header]
            headers += o_header if isinstance(o_header, list) else [o_header]
            headers += a_header if isinstance(a_header, list) else [a_header]
            headers = [h for h in list(set(headers)) if h != '']

            if len(val_qas) > args.max_val_samples_for_optimization:
                np.random.seed(DEFAULT_SEED)
                subsample_idxs = np.random.choice(len(val_qas), args.max_val_samples_for_optimization, replace=False)
                val_qas = [val_qas[idx] for idx in subsample_idxs]
                val_qa_dict = [val_qa_dict[idx] for idx in subsample_idxs]

            if args.debug:
                val_qa_dict = val_qa_dict[:5]
                val_qas = val_qas[:5]

            if args.verbose:
                print(f'System Prompt Seed: {"default" if system_prompt_seed is None else system_prompt_seed}')
                print(f'Template Seed: {"default" if template_seed is None else template_seed}')
                print(f'Few-Shot Example Seed: {"n/a" if few_shot_seed is None else few_shot_seed}')
                print(f'### Sample Input Prompt ###\n\n{val_qas[0]}')

            val_results = _infer(val_qas, val_qa_dict, add_stop_words=headers)
            _, val_acc = evaluate_accuracy_exact_match(val_results)
            val_acc_dict['system_prompt_seed'].append('default' if system_prompt_seed is None else system_prompt_seed)
            val_acc_dict['prompt_template_seed'].append('default' if template_seed is None else template_seed)
            val_acc_dict['few_shot_seed'].append('n/a' if few_shot_seed is None else few_shot_seed)
            val_acc_dict['acc'].append(val_acc)

            logging.info(f'Val. Accuracy: {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_system_prompt = val_dataset.system_prompt
                best_template = val_dataset.prompt_template
                best_few_shot_seed = sample_kwargs['few_shot_seed']

        system_prompt_to_use = best_system_prompt
        prompt_template_to_use = best_template
        few_shot_seed_to_use = best_few_shot_seed

    elif args.use_optimized_prompt:
        # Load the prompt template optimized for the given model
        logging.info('Using the prompt template optimized for the given model...')
        val_acc_df = pd.read_csv(osp.join(args.log_dir, 'val_accs.csv'))
        best_config = val_acc_df[val_acc_df['acc'] == val_acc_df['acc'].max()].iloc[0]

        system_prompt_to_use = get_llm_system_prompt(args.model.name, args.dataset.name)

        best_prompt_template_seed = best_config['prompt_template_seed']
        if best_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_llm_prompt_template(seed=int(best_prompt_template_seed))

        best_few_shot_seed = best_config['few_shot_seed']
        if best_few_shot_seed == 'n/a' or np.isnan(best_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(best_few_shot_seed)

    elif args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        # Load the prompt template optimized for the medical model
        logging.info('Using the prompt template optimized for the medical model...')
        med_model_name = GENERAL_MODEL_TO_MEDICAL_MODEL[args.model.name]
        if isinstance(med_model_name, list): # If a general-domain LLM is the base model for multiple medical LLMs
            if args.med_model_name is not None and args.med_model_name != '':
                med_model_name = args.med_model_name
            else:
                med_model_name = med_model_name[0]

        logging.info(f'Medical Model: "{med_model_name}"')
        med_log_dir = args.log_dir.replace(args.model.name, med_model_name)
        med_val_acc_df = pd.read_csv(osp.join(med_log_dir, 'val_accs.csv'))
        med_best_config = med_val_acc_df[med_val_acc_df['acc'] == med_val_acc_df['acc'].max()].iloc[0]

        system_prompt_to_use = get_llm_system_prompt(med_model_name, args.dataset.name)

        med_prompt_template_seed = med_best_config['prompt_template_seed']
        if med_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_llm_prompt_template(seed=int(med_prompt_template_seed))

        med_few_shot_seed = med_best_config['few_shot_seed']
        if med_few_shot_seed == 'n/a' or np.isnan(med_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(med_few_shot_seed)

        # Update the prompt model name to be the medical model
        prompt_model_name = med_model_name

        if args.model.name == 'llama-3-8b':
            tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

        elif args.model.name == 'llama-2-70b' and med_model_name == 'clinical-camel-70b':
            tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

    else:
        # Otherwise, use the default
        system_prompt_to_use = None
        prompt_template_to_use = None
        few_shot_seed_to_use = DEFAULT_SEED

    # Load test dataset
    logging.info(f'Evaluating performance on {args.eval_split} data...')
    if args.prompt_type == 'few-shot':
        try:
            assert(args.eval_split != 'train')
        except:
            logging.error('Few-shot evaluation is only supported on the validation and test sets.')
            raise ValueError

        splits = ['train', args.eval_split]
        main_split = args.eval_split
    else:
        splits = [args.eval_split]
        main_split = args.eval_split
    
    test_dataset = dataset_cls(splits=splits, main_split=main_split, **dataset_args)

    if args.prompt_type == 'few-shot':
        test_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed_to_use)
    
    test_dataset.load_and_apply_prompt_template(
        model_name=prompt_model_name,
        prompt_type=args.prompt_type,
        tokenize=False,
        tokenizer=tokenizer,
        system_prompt=system_prompt_to_use,
        prompt_template=prompt_template_to_use
    )
    test_qas = test_dataset.qas[args.eval_split]
    test_qa_dict = test_dataset.qa_dict[args.eval_split]

    # Add question, options, and answer headers to stop word list to avoid repetitive completions in the output
    prompt_template_str = test_dataset.prompt_template['template_str']
    q_header = prompt_template_str['q_header']
    o_header = prompt_template_str['o_header']
    a_header = prompt_template_str['a_header']
    
    headers = []
    headers += q_header if isinstance(q_header, list) else [q_header]
    headers += o_header if isinstance(o_header, list) else [o_header]
    headers += a_header if isinstance(a_header, list) else [a_header]
    headers = [h for h in list(set(headers)) if h != '']

    if args.debug:
        test_qa_dict = test_qa_dict[:5]
        test_qas = test_qas[:5]

    if args.verbose:
        print(f'### Sample Input Prompt ###\n\n{test_qas[0]}')

    results = _infer(test_qas, test_qa_dict, add_stop_words=headers)
    test_evals, test_acc = evaluate_accuracy_exact_match(results, verbose=args.verbose)

    if args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        med_model_name = GENERAL_MODEL_TO_MEDICAL_MODEL[args.model.name]
        if isinstance(med_model_name, list): # If a general-domain LLM is the base model for multiple medical LLMs
            if args.med_model_name is not None and args.med_model_name != '':
                med_model_name = args.med_model_name
            else:
                med_model_name = med_model_name[0]
            
            results[f'outputs_med_{med_model_name}'] = results.pop('outputs')
            results[f'confidences_med_{med_model_name}'] = results.pop('confidences')
            results[f'pred_confidences_med_{med_model_name}'] = results.pop('pred_confidences')
            results[f'evals_med_{med_model_name}'] = test_evals
            results[f'accuracy_med_{med_model_name}'] = test_acc
        else:
            results['outputs_med'] = results.pop('outputs')
            results['confidences_med'] = results.pop('confidences')
            results['pred_confidences_med'] = results.pop('pred_confidences')
            results['evals_med'] = test_evals
            results['accuracy_med'] = test_acc
    else:
        results['evals'] = test_evals
        results['accuracy'] = test_acc

    logging.info(f'Test Accuracy (Mean): {test_acc:.4f}')

    if accelerator.is_main_process:
        if args.optimize_prompt:
            # Save template
            template_str = prompt_template_to_use['template_str']
            template_str_path = osp.join(args.log_dir, 'template_str.yaml')
            with open(template_str_path, 'w') as fh:
                yaml.dump(template_str, fh)

            template_path = osp.join(args.log_dir, 'template.pkl')
            with open(template_path, 'wb') as fh:
                dill.dump(template_path, fh)

            # Save validation accuracies if prompt was optimized
            val_acc_path = osp.join(args.log_dir, 'val_accs.csv')
            val_acc_df = pd.DataFrame(val_acc_dict)
            val_acc_df.to_csv(val_acc_path, index=False)

        # Save results
        results_path = osp.join(args.log_dir, f'{args.eval_split}_results.pkl')
        
        if osp.exists(results_path):
            orig_results = pickle.load(open(results_path, 'rb'))
            results = orig_results | results
        
        with open(results_path, 'wb') as fh:
            pickle.dump(results, fh)

        logging.info(f'Results saved in "{results_path}".')
        logging.info(f'Time elapsed: {str(datetime.now() - start_time)}.')

def main_accelerate_single_process(args):
    '''Main LLM inference function based on accelerate (single process).'''

    start_time = datetime.now()
    
    accelerator = Accelerator()
    device = accelerator.device

    # Quantization config
    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type
        )
    else:
        quantization_config = None
    
    # Load model and tokenizer
    if 'lora' in args.model.name:
        adapter_config_path = osp.join(args.model.path, 'adapter_config.json')
        adapter_config = json.load(open(adapter_config_path))
        base_model_path = adapter_config['base_model_name_or_path']

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            cache_dir=args.paths.hf_cache_dir,
            quantization_config=quantization_config,
            attn_implementation=args.attn_implementation,
            torch_dtype=(torch.bfloat16 if args.bf16 else None),
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model.path,
            cache_dir=args.paths.hf_cache_dir,
            use_fast=True
        )

        # Check if there is a mismatch between tokenizer vocab size and embedding matrix
        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            logging.warning('Mismatch between vocab size and embedding matrix.')
            model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, args.model.path)
        logging.info('LoRA weights loaded.')
    else:
        base_model_path = args.model.path
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            cache_dir=args.paths.hf_cache_dir,
            quantization_config=quantization_config,
            attn_implementation=args.attn_implementation,
            torch_dtype=(torch.bfloat16 if args.bf16 else None),
            low_cpu_mem_usage=True,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=args.paths.hf_cache_dir,
            use_fast=True
        )

    tokenizer.padding_side = 'left'

    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if getattr(tokenizer, 'model_max_length', None) is None or tokenizer.model_max_length == VERY_LARGE_INTEGER:
        tokenizer.model_max_length = model.config.max_position_embeddings

    # Adjust context window config for Med42-v1-70B
    # NOTE: Model card mentions that context window size is 4k, but the config file 
    if 'med42-70b' in base_model_path.lower() and 'llama3' not in base_model_path.lower():
        tokenizer.model_max_length = 4096

    # ClinicalCamel: Add Llama-2 tokenizer chat template
    if 'camel' in base_model_path.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

    # OpenBioLLM: Add Llama-3 tokenizer chat template
    elif 'openbiollm' in base_model_path.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    # Mistral: Adjust chat template
    elif 'mistral-7b' in base_model_path.lower() and 'biomistral' not in base_model_path.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    
    model.eval()
    accelerator.wait_for_everyone()

    def _infer(qas, raw_qas, add_stop_words=None):
        questions, options, answers = zip(*raw_qas)
        output_texts = [] # Generated texts
        confidences = [] # Full softmax scores over options
        pred_confidences = [] # Confidence scores for the predicted option

        pbar = tqdm(enumerate(qas), disable=(not accelerator.is_main_process), total=len(qas))
        for idx, prompt in pbar:
            tokenized = tokenizer(prompt, return_tensors='pt')
            input_ids = tokenized.input_ids.to(device)
            attn_masks = tokenized.attention_mask.to(device)
            
            # NOTE: eos_token_id is excluded from bad_words_ids if present, when generating tokens
            # Reference: https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/generation/logits_process.py#L1253
            # Reference: https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/generation/logits_process.py#L1044
            # Passing bad_words_ids = adding -inf bias to the logits corresponding to the specified token IDs
            if args.constrain_vocab:
                whitelist = [tokenizer.convert_tokens_to_ids(chr(ord('A')+i)) for i in range(len(options[idx]))]
                bad_words_ids = [[i] for i in range(len(tokenizer)) if i not in whitelist]
            else:
                bad_words_ids = None

            seeds = [DEFAULT_SEED] if args.temperature == 0 else np.arange(args.n_seeds)
            do_sample = not (args.temperature == 0)
            
            qa_output_texts = []
            qa_confidences = []
            qa_pred_confidences = []
            for seed in seeds:
                transformers.set_seed(seed)

                stop_words = [tokenizer.decode(tokenizer.eos_token_id), '###']
                if any([m in args.model.name for m in ['llama-3', 'openbiollm', 'med42-v2']]):
                    stop_words.append('<|eot_id|>')

                if add_stop_words is not None:
                    stop_words += add_stop_words

                stop_criteria = KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)

                # Generate output tokens
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attn_masks,
                        do_sample=do_sample,
                        temperature=(args.temperature if do_sample else None),
                        top_p=(args.top_p if do_sample else None),
                        num_beams=args.num_beams,
                        max_new_tokens=(1 if args.constrain_vocab else args.max_new_tokens),
                        use_cache=True,
                        bad_words_ids=bad_words_ids,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.pad_token_id,
                        stopping_criteria=[stop_criteria]
                    )
            
                # Decode output tokens
                input_token_len = input_ids.shape[1]
                output_ids = outputs.sequences[:,input_token_len:]
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                output_text = output_text.strip()

                for stop_word in stop_words:
                    if output_text.endswith(stop_word):
                        output_text = output_text[:-len(stop_word)]
                        break
            
                # Constrain the output vocabulary
                if args.constrain_vocab:
                    logits = outputs.scores[0].squeeze(0).clone()
                    eos_bias = torch.zeros_like(logits)
                    eos_bias[tokenizer.eos_token_id] = -torch.inf
                    logits += eos_bias
                    probs = torch.softmax(logits, dim=0)
                    option_idxs = [tokenizer.convert_tokens_to_ids(chr(ord('A')+i)) for i in range(len(options[idx]))]
                    option_probs = probs[option_idxs]
                    Z = torch.sum(option_probs)
                    option_probs /= Z # Normalize

                    # Take the argmax of token probabilities
                    if args.temperature == 0:
                        output_idx = torch.argmax(option_probs)
                        output_text = chr(ord('A') + output_idx)
                        confidence = torch.max(option_probs).item()
                    
                    # Sample from the constrained vocabulary
                    elif args.temperature > 0:
                        torch.manual_seed(seed)
                        output_idx = torch.multinomial(option_probs, 1).item()
                        output_text = chr(ord('A') + output_idx)
                        confidence = option_probs[output_idx].item()

                    qa_confidences.append(option_probs.cpu().numpy())
                    qa_pred_confidences.append(confidence)

                qa_output_texts.append(output_text)

            output_texts.append(qa_output_texts)
            confidences.append(qa_confidences)
            pred_confidences.append(qa_pred_confidences)

        results = dict(
            questions=questions, 
            outputs=gather_object(output_texts), 
            answers=answers, 
            options=options, 
            confidences=gather_object(confidences),
            pred_confidences=gather_object(pred_confidences)
        )

        return results

    # Default dataset arguments
    dataset_cls = DATASET_NAME_TO_CLS[args.dataset.name]
    dataset_args = dict(
        name=args.dataset.name,
        qa_dir=args.dataset.path,
        hf_cache_dir=args.paths.hf_cache_dir
    )

    # Name of the model to use for fetching the prompt template for final evaluation
    prompt_model_name = args.model.name

    # Optimize the prompt based on validation performance
    if args.optimize_prompt:
        logging.info('Running prompt optimization...')
        splits = ['train', 'val'] if args.prompt_type == 'few-shot' else ['val']
        main_split = 'val'

        # Load validation dataset
        val_dataset = dataset_cls(splits=splits, main_split=main_split, **dataset_args)

        # Keep track of best template
        val_acc_dict = {
            'system_prompt_seed': [], 
            'prompt_template_seed': [], 
            'few_shot_seed': [], 
            'acc': []
        }
        best_system_prompt = None
        best_template = None
        best_few_shot_seed = None
        best_val_acc = -np.inf

        # Randomly sample prompts
        if args.model.name in COMPONENTS_TO_FIX_FOR_SAMPLING.keys():
            fix_system_prompt = 'system_prompt' in COMPONENTS_TO_FIX_FOR_SAMPLING[args.model.name]
            system_prompt_seeds = [None] if fix_system_prompt else [None] + list(range(args.n_system_prompt_seeds))
            
            fix_prompt_template = 'prompt_template' in COMPONENTS_TO_FIX_FOR_SAMPLING[args.model.name]
            prompt_template_seeds = [None] if fix_prompt_template else [None] + list(range(args.n_prompt_template_seeds))
        else:
            system_prompt_seeds = [None] + list(range(args.n_system_prompt_seeds))
            prompt_template_seeds = [None] + list(range(args.n_prompt_template_seeds))

        few_shot_seeds = list(range(args.n_few_shot_seeds)) if args.prompt_type == 'few-shot' else [DEFAULT_SEED]

        sample_kwargs_list = [
            dict(system_prompt_seed=s1, prompt_template_seed=s2, few_shot_seed=s3)
            for s1 in system_prompt_seeds
            for s2 in prompt_template_seeds
            for s3 in few_shot_seeds
        ]

        for sample_kwargs in sample_kwargs_list:
            system_prompt_seed = sample_kwargs['system_prompt_seed']
            template_seed = sample_kwargs['prompt_template_seed']
            few_shot_seed = sample_kwargs['few_shot_seed']
            
            if args.prompt_type == 'few-shot':
                val_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed)
            
            val_dataset.load_and_apply_prompt_template(
                model_name=args.model.name,
                prompt_type=args.prompt_type,
                tokenize=False,
                tokenizer=tokenizer,
                sample_kwargs=sample_kwargs
            )
            val_qas = val_dataset.qas['val']
            val_qa_dict = val_dataset.qa_dict['val']

            # Add question, options, and answer headers to stop word list to avoid repetitive completions in the output
            prompt_template_str = val_dataset.prompt_template['template_str']
            q_header = prompt_template_str['q_header']
            o_header = prompt_template_str['o_header']
            a_header = prompt_template_str['a_header']
            
            headers = []
            headers += q_header if isinstance(q_header, list) else [q_header]
            headers += o_header if isinstance(o_header, list) else [o_header]
            headers += a_header if isinstance(a_header, list) else [a_header]
            headers = [h for h in list(set(headers)) if h != '']

            if len(val_qas) > args.max_val_samples_for_optimization:
                np.random.seed(DEFAULT_SEED)
                subsample_idxs = np.random.choice(len(val_qas), args.max_val_samples_for_optimization, replace=False)
                val_qas = [val_qas[idx] for idx in subsample_idxs]
                val_qa_dict = [val_qa_dict[idx] for idx in subsample_idxs]

            if args.debug:
                val_qa_dict = val_qa_dict[:5]
                val_qas = val_qas[:5]

            if args.verbose:
                print(f'System Prompt Seed: {"default" if system_prompt_seed is None else system_prompt_seed}')
                print(f'Template Seed: {"default" if template_seed is None else template_seed}')
                print(f'Few-Shot Example Seed: {"n/a" if few_shot_seed is None else few_shot_seed}')
                print(f'### Sample Input Prompt ###\n\n{val_qas[0]}')

            val_results = _infer(val_qas, val_qa_dict, add_stop_words=headers)
            _, val_acc = evaluate_accuracy_exact_match(val_results)
            val_acc_dict['system_prompt_seed'].append('default' if system_prompt_seed is None else system_prompt_seed)
            val_acc_dict['prompt_template_seed'].append('default' if template_seed is None else template_seed)
            val_acc_dict['few_shot_seed'].append('n/a' if few_shot_seed is None else few_shot_seed)
            val_acc_dict['acc'].append(val_acc)

            logging.info(f'Val. Accuracy: {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_system_prompt = val_dataset.system_prompt
                best_template = val_dataset.prompt_template
                best_few_shot_seed = sample_kwargs['few_shot_seed']

        system_prompt_to_use = best_system_prompt
        prompt_template_to_use = best_template
        few_shot_seed_to_use = best_few_shot_seed
    
    elif args.use_optimized_prompt:
        # Load the prompt template optimized for the given model
        logging.info('Using the prompt template optimized for the given model...')
        val_acc_df = pd.read_csv(osp.join(args.log_dir, 'val_accs.csv'))
        best_config = val_acc_df[val_acc_df['acc'] == val_acc_df['acc'].max()].iloc[0]

        system_prompt_to_use = get_llm_system_prompt(args.model.name, args.dataset.name)

        best_prompt_template_seed = best_config['prompt_template_seed']
        if best_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_llm_prompt_template(seed=int(best_prompt_template_seed))

        best_few_shot_seed = best_config['few_shot_seed']
        if best_few_shot_seed == 'n/a' or np.isnan(best_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(best_few_shot_seed)
        
    elif args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        # Load the prompt template optimized for the medical model
        logging.info('Using the prompt template optimized for the medical model...')
        med_model_name = GENERAL_MODEL_TO_MEDICAL_MODEL[args.model.name]
        if isinstance(med_model_name, list): # If a general-domain LLM is the base model for multiple medical LLMs
            if args.med_model_name is not None and args.med_model_name != '':
                med_model_name = args.med_model_name
            else:
                med_model_name = med_model_name[0]
        
        logging.info(f'Medical Model: "{med_model_name}"')
        med_log_dir = args.log_dir.replace(args.model.name, med_model_name)
        med_val_acc_df = pd.read_csv(osp.join(med_log_dir, 'val_accs.csv'))
        med_best_config = med_val_acc_df[med_val_acc_df['acc'] == med_val_acc_df['acc'].max()].iloc[0]
        
        system_prompt_to_use = get_llm_system_prompt(med_model_name, args.dataset.name)

        med_prompt_template_seed = med_best_config['prompt_template_seed']
        if med_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_llm_prompt_template(seed=int(med_prompt_template_seed))

        med_few_shot_seed = med_best_config['few_shot_seed']
        if med_few_shot_seed == 'n/a' or np.isnan(med_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(med_few_shot_seed)

        # Update the prompt model name to be the medical model
        prompt_model_name = med_model_name

        if args.model.name == 'llama-3-8b':
            tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

        elif args.model.name == 'llama-2-70b' and med_model_name == 'clinical-camel-70b':
            tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
    
    else:
        # Otherwise, use the default
        system_prompt_to_use = None
        prompt_template_to_use = None
        few_shot_seed_to_use = DEFAULT_SEED

    # Load test dataset
    logging.info('Evaluating performance on test data...')
    if args.prompt_type == 'few-shot':
        try:
            assert(args.eval_split != 'train')
        except:
            logging.error('Few-shot evaluation is only supported on the validation and test sets.')
            raise ValueError

        splits = ['train', args.eval_split]
        main_split = args.eval_split
    else:
        splits = [args.eval_split]
        main_split = args.eval_split
    
    test_dataset = dataset_cls(splits=splits, main_split=main_split, **dataset_args)

    if args.prompt_type == 'few-shot':
        test_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed_to_use)
    
    test_dataset.load_and_apply_prompt_template(
        model_name=prompt_model_name,
        prompt_type=args.prompt_type,
        tokenize=False,
        tokenizer=tokenizer,
        system_prompt=system_prompt_to_use,
        prompt_template=prompt_template_to_use
    )
    test_qas = test_dataset.qas[args.eval_split]
    test_qa_dict = test_dataset.qa_dict[args.eval_split]

    # Add question, options, and answer headers to stop word list to avoid repetitive completions in the output
    prompt_template_str = test_dataset.prompt_template['template_str']
    q_header = prompt_template_str['q_header']
    o_header = prompt_template_str['o_header']
    a_header = prompt_template_str['a_header']
    
    headers = []
    headers += q_header if isinstance(q_header, list) else [q_header]
    headers += o_header if isinstance(o_header, list) else [o_header]
    headers += a_header if isinstance(a_header, list) else [a_header]
    headers = [h for h in list(set(headers)) if h != '']

    if args.debug:
        test_qa_dict = test_qa_dict[:5]
        test_qas = test_qas[:5]

    if args.verbose:
        print(f'### Sample Input Prompt ###\n\n{test_qas[0]}')

    results = _infer(test_qas, test_qa_dict, add_stop_words=headers)
    test_evals, test_acc = evaluate_accuracy_exact_match(results, verbose=args.verbose)
    
    if args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        med_model_name = GENERAL_MODEL_TO_MEDICAL_MODEL[args.model.name]
        if isinstance(med_model_name, list): # If a general-domain LLM is the base model for multiple medical LLMs
            if args.med_model_name is not None and args.med_model_name != '':
                med_model_name = args.med_model_name
            else:
                med_model_name = med_model_name[0]
            
            results[f'outputs_med_{med_model_name}'] = results.pop('outputs')
            results[f'confidences_med_{med_model_name}'] = results.pop('confidences')
            results[f'pred_confidences_med_{med_model_name}'] = results.pop('pred_confidences')
            results[f'evals_med_{med_model_name}'] = test_evals
            results[f'accuracy_med_{med_model_name}'] = test_acc
        else:
            results['outputs_med'] = results.pop('outputs')
            results['confidences_med'] = results.pop('confidences')
            results['pred_confidences_med'] = results.pop('pred_confidences')
            results['evals_med'] = test_evals
            results['accuracy_med'] = test_acc
    else:
        results['evals'] = test_evals
        results['accuracy'] = test_acc

    logging.info(f'Test Accuracy (Mean): {test_acc:.4f}')

    if accelerator.is_main_process:
        if args.optimize_prompt:
            # Save template
            template_str = prompt_template_to_use['template_str']
            template_str_path = osp.join(args.log_dir, 'template_str.yaml')
            with open(template_str_path, 'w') as fh:
                yaml.dump(template_str, fh)

            template_path = osp.join(args.log_dir, 'template.pkl')
            with open(template_path, 'wb') as fh:
                dill.dump(template_path, fh)

            # Save validation accuracies if prompt was optimized
            val_acc_path = osp.join(args.log_dir, 'val_accs.csv')
            val_acc_df = pd.DataFrame(val_acc_dict)
            val_acc_df.to_csv(val_acc_path, index=False)

        # Save results
        results_path = osp.join(args.log_dir, f'{args.eval_split}_results.pkl')
        
        if osp.exists(results_path):
            orig_results = pickle.load(open(results_path, 'rb'))
            results = orig_results | results
        
        with open(results_path, 'wb') as fh:
            pickle.dump(results, fh)

        logging.info(f'Results saved in "{results_path}".')
        logging.info(f'Time elapsed: {str(datetime.now() - start_time)}.')

def main_vllm(args):
    '''
        Main LLM inference function based on vLLM; generally much faster than using accelerate.

        NOTE: Only applicable for LoRA rank <= 64.

    '''

    start_time = datetime.now()
    
    # Load (base) model
    if 'lora' in args.model.name:
        adapter_config_path = osp.join(args.model.path, 'adapter_config.json')
        adapter_config = json.load(open(adapter_config_path))
        base_model_path = adapter_config['base_model_name_or_path']
        max_lora_rank = adapter_config['r']
    else:
        base_model_path = args.model.path
        max_lora_rank = 16 # Default

    model_config = AutoConfig.from_pretrained(
        base_model_path,
        cache_dir=args.paths.hf_cache_dir
    )
    vocab_size = AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=args.paths.hf_cache_dir
    ).vocab_size
    
    model = LLM(
        model=base_model_path, 
        tokenizer=args.model.path,
        download_dir=args.paths.hf_cache_dir,
        tensor_parallel_size=len(args.gpu_ids),
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_lora=('lora' in args.model.name),
        max_lora_rank=max_lora_rank,
        max_logprobs=vocab_size
    )
    
    lora_request = LoRARequest('adapters', 1, args.model.path) if 'lora' in args.model.name else None

    # Load tokenizer
    tokenizer = model.get_tokenizer()

    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if getattr(tokenizer, 'model_max_length', None) is None or tokenizer.model_max_length == VERY_LARGE_INTEGER:
        tokenizer.model_max_length = model_config.max_position_embeddings

    # Adjust context window config for Med42-v1-70B
    # NOTE: Model card mentions that context window size is 4k, but the config file 
    if 'med42-70b' in base_model_path.lower() and 'llama3' not in base_model_path.lower():
        tokenizer.model_max_length = 4096

    # ClinicalCamel: Add Llama-2 tokenizer chat template
    if 'camel' in base_model_path.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

    # OpenBioLLM: Add Llama-3 tokenizer chat template
    elif 'openbiollm' in base_model_path.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    # Mistral: Adjust chat template
    elif 'mistral-7b' in base_model_path.lower() and 'biomistral' not in base_model_path.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    # Set up LLM generation config
    n_seeds = 1 if args.temperature == 0 else args.n_seeds

    if args.constrain_vocab:
        logprobs = tokenizer.vocab_size
    else:
        logprobs = None # Don't get logprobs

    stop_words = [tokenizer.decode(tokenizer.eos_token_id), '###']
    if any([m in args.model.name for m in ['llama-3', 'openbiollm', 'med42-v2']]):
        stop_words.append('<|eot_id|>')

    # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py#L61
    # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L63
    # NOTE: bad_words_ids (and therefore, constrain_vocab) is not supported by vLLM.
    sampling_params = SamplingParams(
        n=n_seeds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=(1 if args.constrain_vocab else args.max_new_tokens),
        logprobs=logprobs,
        skip_special_tokens=True,
        stop=stop_words,
        seed=DEFAULT_SEED
    )

    # Helper function for running inference
    def _infer(
        qas, 
        raw_qas, 
        sampling_params,
        constrain_vocab=False,
        lora_request=None
    ):
        outputs = model.generate(qas, sampling_params, lora_request=lora_request)
        output_texts = [[o.text.strip() for o in output.outputs] for output in outputs]
        confidences = []
        pred_confidences = []

        # Check if dataset is multiple-choice
        is_mcq = len(raw_qas[0]) > 2

        # Predict with first-token log-probabilities
        if is_mcq:
            questions, options, answers = zip(*raw_qas)
        else:
            questions, answers = zip(*raw_qas)
            options = None

        for i, output in enumerate(outputs): # Number of inputs
            qa_confidences = []
            qa_pred_confidences = []

            # NOTE: Constrained decoding via logit bias is not directly supported by vLLM.
            # We therefore take the token log-probabilities to manually constrain the token output.
            if constrain_vocab:
                for j, o in enumerate(output.outputs): # Number of samples (i.e., seeds)
                    try:
                        first_token_logprobs = o.logprobs[0]
                    except:
                        logging.warning(f'No first-token logprob found; output text: {output_texts[i][j]}')
                        qa_confidences.append(np.array([]))
                        qa_pred_confidences.append(np.nan)
                        output_texts[i][j] = ''
                        continue
                    
                    n_options = len(options[i])
                    option_chrs = [chr(ord('A')+k) for k in range(n_options)]
                    option_ids = [tokenizer.convert_tokens_to_ids(c) for c in option_chrs]
                    option_logprobs = []
                    for option_id in option_ids:
                        try:
                            option_logprobs.append(first_token_logprobs[option_id].logprob)
                        except:
                            option_logprobs.append(-1e16)

                    option_probs = np.exp(option_logprobs)
                    Z = np.sum(option_probs)
                    option_probs /= Z # Normalize

                    # Take the argmax of token probabilities
                    if args.temperature == 0:
                        output_idx = np.argmax(option_probs)
                        output_text = chr(ord('A') + output_idx)
                        confidence = np.max(option_probs)
                    
                    # Sample from the constrained vocabulary
                    elif args.temperature > 0:
                        np.random.seed(j)
                        output_idx = np.argmax(np.random.multinomial(1, option_probs))
                        output_text = chr(ord('A') + output_idx)
                        confidence = option_probs[output_idx]

                    qa_confidences.append(option_probs)
                    qa_pred_confidences.append(confidence)
                    output_texts[i][j] = output_text 

            confidences.append(qa_confidences)
            pred_confidences.append(qa_pred_confidences)
            
        results = dict(
            questions=questions,
            outputs=output_texts,
            answers=answers,
            options=options,
            confidences=confidences,
            pred_confidences=pred_confidences
        )

        return results

    # Default dataset arguments
    dataset_cls = DATASET_NAME_TO_CLS[args.dataset.name]
    dataset_args = dict(
        name=args.dataset.name,
        qa_dir=args.dataset.path,
        hf_cache_dir=args.paths.hf_cache_dir
    )

    # Name of the model to use for fetching the prompt template for final evaluation
    prompt_model_name = args.model.name

    # Optimize the prompt based on validation performance
    if args.optimize_prompt:
        logging.info('Running prompt optimization...')
        splits = ['train', 'val'] if args.prompt_type == 'few-shot' else ['val']
        main_split = 'val'

        # Load validation dataset
        val_dataset = dataset_cls(
            splits=splits,
            main_split=main_split,
            **dataset_args
        )

        best_system_prompt = None
        best_template = None
        best_few_shot_seed = None

        # Keep track of best template
        val_acc_dict = {
            'system_prompt_seed': [], 
            'prompt_template_seed': [], 
            'few_shot_seed': [], 
            'acc': []
        }
        best_val_acc = -np.inf

        # Randomly sample prompts
        if args.model.name in COMPONENTS_TO_FIX_FOR_SAMPLING.keys():
            fix_system_prompt = 'system_prompt' in COMPONENTS_TO_FIX_FOR_SAMPLING[args.model.name]
            system_prompt_seeds = [None] if fix_system_prompt else [None] + list(range(args.n_system_prompt_seeds))
            
            fix_prompt_template = 'prompt_template' in COMPONENTS_TO_FIX_FOR_SAMPLING[args.model.name]
            prompt_template_seeds = [None] if fix_prompt_template else [None] + list(range(args.n_prompt_template_seeds))
        else:
            system_prompt_seeds = [None] + list(range(args.n_system_prompt_seeds))
            prompt_template_seeds = [None] + list(range(args.n_prompt_template_seeds))

        few_shot_seeds = list(range(args.n_few_shot_seeds)) if args.prompt_type == 'few-shot' else [None]

        sample_kwargs_list = [
            dict(system_prompt_seed=s1, prompt_template_seed=s2, few_shot_seed=s3)
            for s1 in system_prompt_seeds
            for s2 in prompt_template_seeds
            for s3 in few_shot_seeds
        ]

        for sample_kwargs in sample_kwargs_list:
            system_prompt_seed = sample_kwargs['system_prompt_seed']
            template_seed = sample_kwargs['prompt_template_seed']
            few_shot_seed = sample_kwargs['few_shot_seed']
            
            if args.prompt_type == 'few-shot':
                val_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed)
            
            val_dataset.load_and_apply_prompt_template(
                model_name=args.model.name,
                prompt_type=args.prompt_type,
                tokenize=False,
                tokenizer=tokenizer,
                sample_kwargs=sample_kwargs
            )
            val_qas = val_dataset.qas['val']
            val_qa_dict = val_dataset.qa_dict['val']

            # Add question, options, and answer headers to stop word list to avoid repetitive completions in the output
            prompt_template_str = val_dataset.prompt_template['template_str']
            q_header = prompt_template_str['q_header']
            o_header = prompt_template_str['o_header']
            a_header = prompt_template_str['a_header']
            
            headers = []
            headers += q_header if isinstance(q_header, list) else [q_header]
            headers += o_header if isinstance(o_header, list) else [o_header]
            headers += a_header if isinstance(a_header, list) else [a_header]
            headers = [h for h in list(set(headers)) if h != '']
            
            val_sampling_params = deepcopy(sampling_params)
            val_sampling_params.stop += headers

            if len(val_qas) > args.max_val_samples_for_optimization:
                np.random.seed(DEFAULT_SEED)
                subsample_idxs = np.random.choice(len(val_qas), args.max_val_samples_for_optimization, replace=False)
                val_qas = [val_qas[idx] for idx in subsample_idxs]
                val_qa_dict = [val_qa_dict[idx] for idx in subsample_idxs]
            
            if args.debug:
                val_qa_dict = val_qa_dict[:5]
                val_qas = val_qas[:5]
        
            if args.verbose:
                print(f'System Prompt Seed: {"default" if system_prompt_seed is None else system_prompt_seed}')
                print(f'Template Seed: {"default" if template_seed is None else template_seed}')
                print(f'Few-Shot Example Seed: {"n/a" if few_shot_seed is None else few_shot_seed}')
                print(f'### Sample Input Prompt ###\n\n{val_qas[0]}')

            val_results = _infer(
                val_qas,
                val_qa_dict,
                val_sampling_params,
                constrain_vocab=args.constrain_vocab,
                lora_request=lora_request
            )
            _, val_acc = evaluate_accuracy_exact_match(val_results)
        
            val_acc_dict['system_prompt_seed'].append('default' if system_prompt_seed is None else system_prompt_seed)
            val_acc_dict['prompt_template_seed'].append('default' if template_seed is None else template_seed)
            val_acc_dict['few_shot_seed'].append('n/a' if few_shot_seed is None else few_shot_seed)
            val_acc_dict['acc'].append(val_acc)

            logging.info(f'Val. Accuracy: {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_system_prompt = val_dataset.system_prompt
                best_template = val_dataset.prompt_template
                best_few_shot_seed = sample_kwargs['few_shot_seed']
    
        system_prompt_to_use = best_system_prompt
        prompt_template_to_use = best_template
        few_shot_seed_to_use = best_few_shot_seed

    elif args.use_optimized_prompt:
        # Load the prompt template optimized for the given model
        logging.info('Using the prompt template optimized for the given model...')
        val_acc_df = pd.read_csv(osp.join(args.log_dir, 'val_accs.csv'))
        best_config = val_acc_df[val_acc_df['acc'] == val_acc_df['acc'].max()].iloc[0]

        system_prompt_to_use = get_llm_system_prompt(args.model.name, args.dataset.name)

        best_prompt_template_seed = best_config['prompt_template_seed']
        if best_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_llm_prompt_template(seed=int(best_prompt_template_seed))

        best_few_shot_seed = best_config['few_shot_seed']
        if best_few_shot_seed == 'n/a' or np.isnan(best_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(best_few_shot_seed)

    elif args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        # Load the prompt template optimized for the medical model
        logging.info('Using the prompt template optimized for the medical model...')
        med_model_name = GENERAL_MODEL_TO_MEDICAL_MODEL[args.model.name]
        if isinstance(med_model_name, list): # If a general-domain LLM is the base model for multiple medical LLMs
            if args.med_model_name is not None and args.med_model_name != '':
                med_model_name = args.med_model_name
            else:
                med_model_name = med_model_name[0]
        
        logging.info(f'Medical Model: "{med_model_name}"')
        med_log_dir = args.log_dir.replace(args.model.name, med_model_name)
        med_val_acc_df = pd.read_csv(osp.join(med_log_dir, 'val_accs.csv'))
        med_best_config = med_val_acc_df[med_val_acc_df['acc'] == med_val_acc_df['acc'].max()].iloc[0]
        
        system_prompt_to_use = get_llm_system_prompt(med_model_name, args.dataset.name)

        med_prompt_template_seed = med_best_config['prompt_template_seed']
        if med_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_llm_prompt_template(seed=int(med_prompt_template_seed))

        med_few_shot_seed = med_best_config['few_shot_seed']
        if med_few_shot_seed == 'n/a' or np.isnan(med_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(med_few_shot_seed)

        # Update the prompt model name to be the medical model
        prompt_model_name = med_model_name

        if args.model.name == 'llama-3-8b':
            tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

        elif args.model.name == 'llama-2-70b' and med_model_name == 'clinical-camel-70b':
            tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE

    else:
        # Otherwise, use the default
        system_prompt_to_use = None
        prompt_template_to_use = None
        few_shot_seed_to_use = None
    
    # Load test dataset
    logging.info('Evaluating performance on test data...')
    if args.prompt_type == 'few-shot':
        try:
            assert(args.eval_split != 'train')
        except:
            logging.error('Few-shot evaluation is only supported on the validation and test sets.')
            raise ValueError

        splits = ['train', args.eval_split]
        main_split = args.eval_split
    else:
        splits = [args.eval_split]
        main_split = args.eval_split
    
    test_dataset = dataset_cls(splits=splits, main_split=main_split, **dataset_args)

    if args.prompt_type == 'few-shot':
        test_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed_to_use)
    
    test_dataset.load_and_apply_prompt_template(
        model_name=prompt_model_name,
        prompt_type=args.prompt_type,
        tokenize=False,
        tokenizer=tokenizer,
        system_prompt=system_prompt_to_use,
        prompt_template=prompt_template_to_use
    )
    test_qas = test_dataset.qas[args.eval_split]
    test_qa_dict = test_dataset.qa_dict[args.eval_split]

    # Add question, options, and answer headers to stop word list to avoid repetitive completions in the output
    prompt_template_str = test_dataset.prompt_template['template_str']
    q_header = prompt_template_str['q_header']
    o_header = prompt_template_str['o_header']
    a_header = prompt_template_str['a_header']
    
    headers = []
    headers += q_header if isinstance(q_header, list) else [q_header]
    headers += o_header if isinstance(o_header, list) else [o_header]
    headers += a_header if isinstance(a_header, list) else [a_header]
    headers = [h for h in list(set(headers)) if h != '']
    
    test_sampling_params = deepcopy(sampling_params)
    test_sampling_params.stop += headers

    if args.debug:
        test_qa_dict = test_qa_dict[:5]
        test_qas = test_qas[:5]

    if args.verbose:
        print(f'### Sample Input Prompt ###\n\n{test_qas[0]}')

    results = _infer(
        test_qas,
        test_qa_dict,
        test_sampling_params,
        constrain_vocab=args.constrain_vocab,
        lora_request=lora_request
    )
    test_evals, test_acc = evaluate_accuracy_exact_match(results, verbose=args.verbose)
    
    if args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        med_model_name = GENERAL_MODEL_TO_MEDICAL_MODEL[args.model.name]
        if isinstance(med_model_name, list): # If a general-domain LLM is the base model for multiple medical LLMs
            if args.med_model_name is not None and args.med_model_name != '':
                med_model_name = args.med_model_name
            else:
                med_model_name = med_model_name[0]
            
            results[f'outputs_med_{med_model_name}'] = results.pop('outputs')
            results[f'confidences_med_{med_model_name}'] = results.pop('confidences')
            results[f'pred_confidences_med_{med_model_name}'] = results.pop('pred_confidences')
            results[f'evals_med_{med_model_name}'] = test_evals
            results[f'accuracy_med_{med_model_name}'] = test_acc
        else:
            results['outputs_med'] = results.pop('outputs')
            results['confidences_med'] = results.pop('confidences')
            results['pred_confidences_med'] = results.pop('pred_confidences')
            results['evals_med'] = test_evals
            results['accuracy_med'] = test_acc
            logging.info(f'Test Accuracy: {test_acc:.4f}')
    else:
        results['evals'] = test_evals
        results['accuracy'] = test_acc
        logging.info(f'Test Accuracy: {test_acc:.4f}')

    if os.environ.get('LOCAL_RANK', '0') == '0':
        if args.optimize_prompt:
            # Save template
            template_str = prompt_template_to_use['template_str']
            template_str_path = osp.join(args.log_dir, 'template_str.yaml')
            with open(template_str_path, 'w') as fh:
                yaml.dump(template_str, fh)

            template_path = osp.join(args.log_dir, 'template.pkl')
            with open(template_path, 'wb') as fh:
                dill.dump(prompt_template_to_use, fh)
            
            val_acc_path = osp.join(args.log_dir, 'val_accs.csv')
            val_acc_df = pd.DataFrame(val_acc_dict)
            val_acc_df.to_csv(val_acc_path, index=False)

        # Save results
        results_path = osp.join(args.log_dir, f'{args.eval_split}_results.pkl')
        
        if osp.exists(results_path):
            orig_results = pickle.load(open(results_path, 'rb'))
            results = orig_results | results
        
        with open(results_path, 'wb') as fh:
            pickle.dump(results, fh)

        logging.info(f'Results saved in "{results_path}".')
        logging.info(f'Time elapsed: {str(datetime.now() - start_time)}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default=None, type=str) # Hydra log directory
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('--use_vllm', default=False, action='store_true')
    parser.add_argument('--use_accelerate_single_process', default=False, action='store_true')
    parsed = parser.parse_args()
    
    with open(parsed.config_path, 'r') as fh:
        args = OmegaConf.create(fh.read()) # Loads omegaconf.DictConfig object

    args.log_dir = parsed.log_dir

    # Log in to HF if required
    if args.model.name in MODELS_REQ_LOGIN:
        huggingface_hub.login(token=args.hf_api_key)

    # Run inference with vLLM if applicable
    if parsed.use_vllm:
        main_vllm(args)
    
    # Run single-process inference with accelerate
    elif parsed.use_accelerate_single_process:
        main_accelerate_single_process(args)
    
    # Run multi-process inference with accelerate
    else:
        main_accelerate(args)