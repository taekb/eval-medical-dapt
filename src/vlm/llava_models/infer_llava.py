import sys
import os
import os.path as osp
import re
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import dill
import yaml
from collections import Counter
from datetime import datetime
from PIL import Image

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

import warnings
warnings.filterwarnings('ignore')

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import CLIPImageProcessor, CLIPVisionModel

from accelerate import Accelerator
from accelerate.utils import gather_object

from omegaconf import OmegaConf

from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model

from llava.constants import (
    IMAGE_TOKEN_INDEX, # -200
    DEFAULT_IMAGE_TOKEN, # <image>
    DEFAULT_IMAGE_PATCH_TOKEN, # <im_patch>
    DEFAULT_IM_START_TOKEN, # <im_start>
    DEFAULT_IM_END_TOKEN, # <im_end>
)

from old_llava import LlavaLlamaForCausalLM as OldLlavaLlamaForCausalLM

from peft import PeftModel

DEFAULT_SEED = 42
SRC_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
VLM_DIR = osp.join(SRC_DIR, 'vlm')
sys.path += [SRC_DIR, VLM_DIR]

from utils import remove_punc, white_space_fix, get_vlm_system_prompt, sample_vlm_prompt_template

from dataset import (
    VQARADDataset,
    PathVQADataset,
    SlakeDataset,
    MMMUDataset
)

DATASET_NAME_TO_CLS = {
    'vqa-rad': VQARADDataset,
    'pvqa': PathVQADataset,
    'slake': SlakeDataset,
    'mmmu_basic-medical-science': MMMUDataset,
    'mmmu_clinical-medicine': MMMUDataset,
    'mmmu_diagnostics-and-laboratory-medicine': MMMUDataset,
    'mmmu_pharmacy': MMMUDataset,
    'mmmu_public-health': MMMUDataset
}
COMPONENTS_TO_FIX_FOR_SAMPLING = {
    'llava-v0-7b': ['system_prompt'],
    'llava-med-7b': ['system_prompt'],
    'open-flamingo-9b': ['system_prompt'],
    'med-flamingo-9b': ['system_prompt']
}
GENERAL_MODEL_TO_MEDICAL_MODEL = {
    'llava-v0-7b': 'llava-med-7b',
    'open-flamingo-9b': 'med-flamingo-9b'
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
        answer_letter = chr(ord('A') + options.index(answer))

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

class LlavaMedKeywordsStoppingCriteria(transformers.StoppingCriteria):
    '''
        Stopping criteria class used in LLaVA-Med.

        Reference: https://github.com/microsoft/LLaVA-Med/blob/main/llava/eval/model_vqa_med.py

    '''

    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

def main(args):
    '''Main inference and evaluation function for LLaVA models.'''

    start_time = datetime.now()

    # Parse model-specific configs
    llava_args = args.model.config

    # Quantization config
    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    bnb_args = {}
    if args.load_in_4bit or args.load_in_8bit:
        bnb_args.update(dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                llm_int8_skip_modules=['mm_projector'],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type
            )
        ))

    disable_torch_init()
    accelerator = Accelerator()
    device = accelerator.device

    # LLaVA-v1.5 models
    if 'llava-v1' in args.model.path:
        # Load the pretrained LLaVA model
        use_flash_attn = (args.attn_implementation == 'flash_attention_2')
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=args.model.path,
            model_base=args.model.base,
            model_name=args.model.name,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            use_flash_attn=use_flash_attn
        )
        model.to(device)

        stop_criteria_class = KeywordsStoppingCriteria
        
        def clean_question(question, mm_use_im_start_end):
            cleaned = question.replace(DEFAULT_IMAGE_TOKEN, '')
            
            if mm_use_im_start_end:
                cleaned = cleaned.replace(DEFAULT_IM_START_TOKEN, '')
                cleaned = cleaned.replace(DEFAULT_IM_END_TOKEN, '')
            
            return cleaned.strip()
        
        def preprocess_image(image_input):
            # No image provided
            if len(image_input) == 0:
                images = None
            else:
                images = []
                for i in image_input:
                    if isinstance(i, str):
                        i = Image.open(i)

                    images.append(image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0])

                images = torch.stack(images).half().cuda()

            return images
        
        def tokenize_input_prompt(prompt):
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()

            return input_ids
    else:
        if 'lora' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model.base, 
                cache_dir=args.paths.hf_cache_dir,
                use_fast=False
            )
            model = OldLlavaLlamaForCausalLM.from_pretrained(
                args.model.base,
                attn_implementation=args.attn_implementation, 
                torch_dtype=compute_dtype, 
                use_cache=True,
                device_map=device,
                cache_dir=args.paths.hf_cache_dir,
                **bnb_args
            )

            non_lora_trainables = torch.load(
                osp.join(args.model.path, 'non_lora_trainables.bin'), 
                map_location='cpu'
            )
            if len(non_lora_trainables) > 0:
                non_lora_trainables = {
                    (k[11:] if k.startswith('base_model.') else k): v
                    for k,v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            logging.info('Loading (Q)LoRA weights...')
            model = PeftModel.from_pretrained(model, args.model.path)
            model = model.merge_and_unload()
        else:
            # Load the pretrained LLaVA model (older implementation)
            tokenizer = AutoTokenizer.from_pretrained(
                args.model.path, 
                cache_dir=args.paths.hf_cache_dir
            )
            model = OldLlavaLlamaForCausalLM.from_pretrained(
                args.model.path, 
                attn_implementation=args.attn_implementation,
                torch_dtype=compute_dtype, 
                use_cache=True,
                device_map=device,
                cache_dir=args.paths.hf_cache_dir,
                **bnb_args
            )

        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, 
            cache_dir=args.paths.hf_cache_dir,
            torch_dtype=compute_dtype
        )
        
        if model.model.vision_tower[0].device.type == 'meta':
            model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(
                model.model.vision_tower[0].config._name_or_path, 
                cache_dir=args.paths.hf_cache_dir,
                device_map=device
            )
            
        model.model.vision_tower[0].to(compute_dtype)
        model.to(device)

        mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_config = model.model.vision_tower[0].config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([
                DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            ])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        stop_criteria_class = LlavaMedKeywordsStoppingCriteria
        
        def clean_question(question, mm_use_im_start_end):
            cleaned = question.replace(DEFAULT_IMAGE_PATCH_TOKEN, '')
            
            if mm_use_im_start_end:
                cleaned = cleaned.replace(DEFAULT_IM_START_TOKEN, '')
                cleaned = cleaned.replace(DEFAULT_IM_END_TOKEN, '')
            
            return cleaned    

        def preprocess_image(image_input):
            # No image provided
            if len(image_input) == 0:
                images = None
            else:
                images = []
                for i in image_input:
                    if isinstance(i, str):
                        i = Image.open(i)

                    images.append(image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0])

                images = torch.stack(images).half().cuda()

            return images
        
        def tokenize_input_prompt(prompt):
            input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).cuda()

            return input_ids

    model.eval()

    # Check conversation template format
    # Reference: https://github.com/haotian-liu/LLaVA/tree/main/llava/conversation.py#L361
    conv_mode = llava_args.conv_mode
    conv = conv_templates[conv_mode].copy()
    sep_style = conv.sep_style.name
    
    # Set up stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2 # e.g., </s>
    keywords = [stop_str]

    def _infer(qas, raw_qas, few_shot_images=None):
        questions, options, answers, images = zip(*raw_qas)
        output_texts = [] # Generated texts
        confidences = [] # Full softmax scores over options
        pred_confidences = [] # Confidence scores for the top option

        with accelerator.split_between_processes(list(zip(range(len(qas)), qas))) as acc_qas:
            pbar = tqdm(acc_qas, disable=(not accelerator.is_main_process), total=len(acc_qas))
            for idx, prompt in pbar:
                input_ids = tokenize_input_prompt(prompt)
                qa_images = few_shot_images + images[idx] if few_shot_images is not None else images[idx]
                qa_images = preprocess_image(qa_images)

                # Limit vocabulary for output generation
                if args.constrain_vocab:
                    # NOTE: tokenizer.vocab_size doesn't include added tokens like DEFAULT_IMAGE_PATCH_TOKEN
                    # Should therefore use len(tokenizer) to completely remove everything except the option tokens.
                    whitelist = [tokenizer.convert_tokens_to_ids(chr(ord('A')+i)) for i in range(len(options[idx]))]
                    bad_words_ids = [[i] for i in range(len(tokenizer)) if i not in whitelist]
                else:
                    bad_words_ids = None

                # Generation config
                seeds = [DEFAULT_SEED] if args.temperature == 0 else np.arange(0,args.n_seeds)

                qa_output_texts = []
                qa_confidences = []
                qa_pred_confidences = []
                for seed in seeds:
                    transformers.set_seed(seed)

                    stop_criteria = stop_criteria_class(keywords, tokenizer, torch.as_tensor(input_ids).cuda())

                    # Generate output tokens
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids,
                            images=qa_images,
                            do_sample=(True if args.temperature > 0 else False),
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=(1 if args.constrain_vocab else args.max_new_tokens),
                            use_cache=True,
                            stopping_criteria=[stop_criteria],
                            bad_words_ids=bad_words_ids,
                            output_scores=True,
                            return_dict_in_generate=True
                        )

                    # Decode output tokens
                    input_token_len = input_ids.shape[1]
                    output_ids = outputs.sequences[:,input_token_len:]
                    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    output_text = output_text.strip()

                    if output_text.endswith(stop_str):
                        output_text = output_text[:-len(stop_str)] # Removing the "</s>" or "###" tag

                    if sep_style == 'SINGLE':
                        output_text = output_text.split(conv.sep)[0].strip()
                        
                    # Contrain the output vocabulary
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

                        qa_confidences.append(option_probs.cpu().float().numpy())
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

    dataset_cls = DATASET_NAME_TO_CLS[args.dataset.name]
    dataset_args = dict(name=args.dataset.name, seed=DEFAULT_SEED, verbose=False)
    if 'mmmu' not in args.dataset.name:
        dataset_args['qa_dir'] = osp.join(args.paths.data_root_dir, args.dataset.qa_dir)
        dataset_args['image_dir'] = osp.join(args.paths.data_root_dir, args.dataset.image_dir)
    else:
        dataset_args['hf_cache_dir'] = args.paths.hf_cache_dir

    llava_kwargs = dict(
        conv_mode=llava_args.conv_mode,
        mm_use_im_start_end=mm_use_im_start_end
    )
    if 'llava-v1' not in args.model.path:
        llava_kwargs['image_token_len'] = image_token_len
    
    # Optimize the few-shot examples based on validation performance
    if args.optimize_prompt:
        if accelerator.is_main_process:
            logging.info('Running few-shot example optimization...')
        
        splits = ['train', 'val'] if args.prompt_type == 'few-shot' else ['val']
        main_split = 'val'

        # Load validation dataset
        val_dataset = dataset_cls(splits=splits, main_split=main_split, **dataset_args)

        # Keep track of best few-shot examples
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

        for i, sample_kwargs in enumerate(sample_kwargs_list):
            if accelerator.is_main_process:
                logging.info(f'Trial: {i+1}/{len(sample_kwargs_list)}...')

            system_prompt_seed = sample_kwargs['system_prompt_seed']
            template_seed = sample_kwargs['prompt_template_seed']
            few_shot_seed = sample_kwargs['few_shot_seed']
            
            if args.prompt_type == 'few-shot':
                val_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed)
            
            val_dataset.load_and_apply_prompt_template(
                model_name=args.model.name,
                prompt_type=args.prompt_type,
                sample_kwargs=sample_kwargs,
                llava_kwargs=llava_kwargs
            )
            val_qas = val_dataset.qas['val']
            val_qa_dict = val_dataset.qa_dict['val']

            if args.prompt_type == 'few-shot':
                few_shot_images = val_dataset.few_shot_images

                if isinstance(few_shot_images[0], list):
                    few_shot_images = [img[0] for img in few_shot_images]
            else:
                few_shot_images = None

            if len(val_qas) > args.max_val_samples_for_optimization:
                np.random.seed(DEFAULT_SEED)
                subsample_idxs = np.random.choice(len(val_qas), args.max_val_samples_for_optimization, replace=False)
                val_qas = [val_qas[idx] for idx in subsample_idxs]
                val_qa_dict = [val_qa_dict[idx] for idx in subsample_idxs]

            if args.debug:
                val_qa_dict = val_qa_dict[:5]
                val_qas = val_qas[:5]

            if args.verbose and accelerator.is_main_process:
                logging.info(f'System Prompt Seed: {"default" if system_prompt_seed is None else system_prompt_seed}')
                logging.info(f'Template Seed: {"default" if template_seed is None else template_seed}')
                logging.info(f'Few-Shot Example Seed: {"n/a" if few_shot_seed is None else few_shot_seed}')
                logging.info(f'### Sample Input Prompt ###\n\n{val_qas[0]}')

            val_results = _infer(val_qas, val_qa_dict, few_shot_images=few_shot_images)
            _, val_acc = evaluate_accuracy_exact_match(val_results, verbose=accelerator.is_main_process)
            val_acc_dict['system_prompt_seed'].append('default' if system_prompt_seed is None else system_prompt_seed)
            val_acc_dict['prompt_template_seed'].append('default' if template_seed is None else template_seed)
            val_acc_dict['few_shot_seed'].append('n/a' if few_shot_seed is None else few_shot_seed)
            val_acc_dict['acc'].append(val_acc)

            if accelerator.is_main_process:
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

        system_prompt_to_use = get_vlm_system_prompt(args.model.name)

        best_prompt_template_seed = best_config['prompt_template_seed']
        if best_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_vlm_prompt_template(seed=int(best_prompt_template_seed))

        best_few_shot_seed = best_config['few_shot_seed']
        if best_few_shot_seed == 'n/a' or np.isnan(best_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(best_few_shot_seed)

    elif args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        # Load the prompt template optimized for the medical model
        logging.info('Using the prompt template optimized for the medical model...')
        med_model_name = GENERAL_MODEL_TO_MEDICAL_MODEL[args.model.name]
        med_log_dir = args.log_dir.replace(args.model.name, med_model_name)
        med_val_acc_df = pd.read_csv(osp.join(med_log_dir, 'val_accs.csv'))
        med_best_config = med_val_acc_df[med_val_acc_df['acc'] == med_val_acc_df['acc'].max()].iloc[0]
        
        system_prompt_to_use = get_vlm_system_prompt(med_model_name)

        med_prompt_template_seed = med_best_config['prompt_template_seed']
        if med_prompt_template_seed == 'default':
            prompt_template_to_use = None
        else:
            prompt_template_to_use = sample_vlm_prompt_template(seed=int(med_prompt_template_seed))

        med_few_shot_seed = med_best_config['few_shot_seed']
        if med_few_shot_seed == 'n/a' or np.isnan(med_few_shot_seed):
            few_shot_seed_to_use = None
        else:
            few_shot_seed_to_use = int(med_few_shot_seed)
    
    else:
        # Otherwise, use the default
        system_prompt_to_use = None
        prompt_template_to_use = None
        few_shot_seed_to_use = DEFAULT_SEED

    # Load test dataset
    if accelerator.is_main_process:
        logging.info('Evaluating performance on test data...')
    
    splits = ['train', 'test'] if args.prompt_type == 'few-shot' else ['test']
    main_split = 'test'
    test_dataset = dataset_cls(splits=splits, main_split=main_split, **dataset_args)

    if args.prompt_type == 'few-shot':
        test_dataset.sample_few_shot_qas(n_shot=args.n_shot, seed=few_shot_seed_to_use)

    test_dataset.load_and_apply_prompt_template(
        model_name=args.model.name,
        prompt_type=args.prompt_type,
        system_prompt=system_prompt_to_use,
        prompt_template=prompt_template_to_use,
        llava_kwargs=llava_kwargs
    )
    test_qas = test_dataset.qas['test']
    test_qa_dict = test_dataset.qa_dict['test']

    if args.prompt_type == 'few-shot':
        few_shot_images = test_dataset.few_shot_images

        if isinstance(few_shot_images[0], list):
            few_shot_images = [img[0] for img in few_shot_images]
    else:
        few_shot_images = None

    if args.debug:
        test_qa_dict = test_qa_dict[:5]
        test_qas = test_qas[:5]

    if args.verbose and accelerator.is_main_process:
        logging.info(f'### Sample Input Prompt ###\n\n{test_qas[0]}')

    results = _infer(test_qas, test_qa_dict, few_shot_images=few_shot_images)
    test_evals, test_acc = evaluate_accuracy_exact_match(results, verbose=accelerator.is_main_process)
    
    if args.use_med_prompt and args.model.name in GENERAL_MODEL_TO_MEDICAL_MODEL.keys():
        results['outputs_med'] = results.pop('outputs')
        results['confidences_med'] = results.pop('confidences')
        results['pred_confidences_med'] = results.pop('pred_confidences')
        results['evals_med'] = test_evals
        results['accuracy_med'] = test_acc
    else:
        results['evals'] = test_evals
        results['accuracy'] = test_acc

    if accelerator.is_main_process:
        logging.info(f'Test Accuracy (Mean): {test_acc:.4f}')

        if args.optimize_prompt:
            # Save template
            template_str = prompt_template_to_use['template_str']
            template_str_path = osp.join(args.log_dir, 'template_str.yaml')
            with open(template_str_path, 'w') as fh:
                yaml.dump(template_str, fh)

            template_path = osp.join(args.log_dir, 'template.pkl')
            with open(template_path, 'wb') as fh:
                dill.dump(prompt_template_to_use, fh)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default=None, type=str) # Hydra log directory
    parser.add_argument('--config_path', default=None, type=str)
    parsed = parser.parse_args()

    with open(parsed.config_path, 'r') as fh:
        args = OmegaConf.create(fh.read()) # Loads omegaconf.DictConfig object

    args.log_dir = parsed.log_dir

    # Run inference and evaluation
    main(args)