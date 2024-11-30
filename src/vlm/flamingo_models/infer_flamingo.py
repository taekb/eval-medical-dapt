import sys
import os
import os.path as osp
import re
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd
import pickle
import dill
import yaml
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
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from accelerate import Accelerator
from accelerate.utils import gather_object

from einops import rearrange
from huggingface_hub import hf_hub_download

from omegaconf import OmegaConf

import open_clip
import open_flamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.flamingo import Flamingo
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.factory import _infer_decoder_layers_attr_name

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

# Adapted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py#L215
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

# Adapted from: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/factory.py#L11
def create_model_and_transforms_with_quant(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: str = None,
    quantization_config: BitsAndBytesConfig = None,
    use_flash_attn: bool = False,
    eval_mode: bool = False,
    **flamingo_kwargs,
):
    '''
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    '''

    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path,
        pretrained=clip_vision_encoder_pretrained,
        cache_dir=cache_dir,
    )
    # Set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=cache_dir,
        use_fast=True
    )
    # Add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    if use_flash_attn:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
            attn_implementation='flash_attention_2'
        )
    else:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            cache_dir=cache_dir,
            quantization_config=quantization_config
        )

    # Hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if 'mpt-1b-redpajama-200b' in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # Convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    if not eval_mode:
        # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
        model.perceiver.requires_grad_(True)
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        if not freeze_lm_embeddings:
            model.lang_encoder.get_input_embeddings().requires_grad_(True)

    logging.info(
        'Flamingo model initialized with {} trainable parameters'.format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    return model, image_processor, text_tokenizer
    
def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    return response.replace('<unk> ', '').strip()

def main(args):
    '''Main inference and evaluation function for Flamingo-style models.'''

    start_time = datetime.now()

    # Parse model-specific configs
    flamingo_args = args.model.config

    # Set up precision
    if args.fp16:
        compute_dtype = torch.float16
    elif args.bf16:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # Set up quantization
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

    accelerator = Accelerator()
    device = accelerator.device

    # Load the pretrained OpenFlamingo model
    model, image_processor, tokenizer = create_model_and_transforms_with_quant(
        clip_vision_encoder_path=flamingo_args.clip_vision_encoder_path,
        clip_vision_encoder_pretrained=flamingo_args.clip_vision_encoder_pretrained,
        lang_encoder_path=flamingo_args.lang_encoder_path,
        tokenizer_path=flamingo_args.tokenizer_path,
        cross_attn_every_n_layers=flamingo_args.cross_attn_every_n_layers,
        cache_dir=args.paths.hf_cache_dir,
        quantization_config=quantization_config,
        use_flash_attn=(args.attn_implementation == 'flash_attention_2'),
        eval_mode=True
    )
    if osp.exists(args.model.path):
        ckpt = torch.load(osp.join(args.model.path, 'checkpoint.pt'), map_location=device)
    else:
        ckpt_path = hf_hub_download(
            repo_id=args.model.path,
            filename=('model.pt' if 'med-flamingo' in args.model.name else 'checkpoint.pt'),
            cache_dir=args.paths.hf_cache_dir
        )
        ckpt = torch.load(ckpt_path, map_location=device)
    
    # Trained checkpoints contain everything including optimizer states.
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    
    # NOTE: _IncompatibleKeys error is expected, since only the Perceiver and x-attention layers in checkpoint.
    model.load_state_dict(ckpt, strict=False)
    model.to(compute_dtype)
    model.to(device)
    model.eval()

    def _infer(qas, raw_qas, few_shot_images=None):
        questions, options, answers, images = zip(*raw_qas)
        output_texts = [] # Generated texts
        confidences = [] # Full softmax scores over options
        pred_confidences = [] # Confidence scores for the top option

        with accelerator.split_between_processes(list(zip(range(len(qas)), qas))) as acc_qas:
            pbar = tqdm(acc_qas, disable=(not accelerator.is_main_process), total=len(acc_qas))
            for idx, prompt in pbar:
                lang_x = tokenizer([prompt], return_tensors='pt')

                # No image provided
                if len(images[idx]) == 0:
                    qa_images = None
                else:
                    qa_images = []
                    for i in images[idx]:
                        if isinstance(i, str):
                            i = Image.open(i).convert('RGB')

                        qa_images.append(i)
                
                if args.prompt_type == 'few-shot' and few_shot_images is not None:
                    qa_images = few_shot_images + qa_images

                # Batch size x num_media x num_frames x channels x height x width
                pixels = [image_processor(im).unsqueeze(0) for im in qa_images]
                pixels = torch.cat(pixels, dim=0)
                pixels = pixels.unsqueeze(1).unsqueeze(0)

                # Limit vocabulary for output generation
                if args.constrain_vocab:
                    whitelist = [tokenizer.convert_tokens_to_ids(chr(ord('A')+i)) for i in range(len(options[idx]))]
                    bad_words_ids = [[i] for i in range(len(tokenizer)) if i not in whitelist]
                else:
                    bad_words_ids = None

                # Override random seed list to the default seed if sampling with zero temperature
                seeds = [DEFAULT_SEED] if args.temperature == 0 else np.arange(0,args.n_seeds)

                # Sample multiple outputs from the model with different random seeds
                qa_output_texts = []
                qa_confidences = []
                qa_pred_confidences = []
                for seed in seeds:
                    transformers.set_seed(seed)

                    # Manually condition the cross-attention layers
                    model.lang_encoder._use_cached_vision_x = True
                    b, T, F = pixels.shape[:3]
                    vision_x = rearrange(pixels, "b T F c h w -> (b T F) c h w")
                    with torch.no_grad():
                        vision_x = model.vision_encoder(vision_x.to(dtype=compute_dtype, device=device))[1]
                    
                    vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
                    vision_x = model.perceiver(vision_x)

                    for layer in model.lang_encoder._get_decoder_layers():
                        layer.condition_vis_x(vision_x)

                    media_locations = lang_x['input_ids'].to(device) == model.media_token_id

                    for layer in model.lang_encoder._get_decoder_layers():
                        layer.condition_media_locations(media_locations)
                        layer.condition_use_cached_media(True)

                    stop_str = ['Question:', tokenizer.decode(model.eoc_token_id)]
                    stop_criteria = KeywordsStoppingCriteria(stop_str, tokenizer, lang_x['input_ids'])

                    # Generate output
                    with torch.inference_mode():
                        outputs = model.generate(
                            vision_x=pixels.to(dtype=compute_dtype, device=device),
                            lang_x=lang_x['input_ids'].to(device),
                            attention_mask=lang_x['attention_mask'].to(device),
                            do_sample=(True if args.temperature > 0 else False),
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=(1 if args.constrain_vocab else args.max_new_tokens),
                            use_cache=True,
                            bad_words_ids=bad_words_ids,
                            output_scores=True,
                            return_dict_in_generate=True,
                            stopping_criteria=[stop_criteria]
                        )

                    # Decode output tokens
                    input_token_len = len(lang_x['input_ids'][0])
                    output_ids = outputs.sequences[:,input_token_len:]
                    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    
                    if 'Question:' in output_text:
                        output_text = output_text.replace('Question:', '')
                    
                    output_text = output_text.strip()

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

    # Optimize the few-shot examples based on validation performance
    if args.optimize_prompt:
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
                sample_kwargs=sample_kwargs
            )
            val_qas = val_dataset.qas['val']
            val_qa_dict = val_dataset.qa_dict['val']

            if args.prompt_type == 'few-shot':
                few_shot_images = val_dataset.few_shot_images

                if isinstance(few_shot_images[0], list):
                    few_shot_images = [img[0] for img in few_shot_images]

                if isinstance(few_shot_images[0], str):
                    few_shot_images = [Image.open(img).convert('RGB') for img in few_shot_images]
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
            _, val_acc = evaluate_accuracy_exact_match(val_results)
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
        prompt_template=prompt_template_to_use
    )
    test_qas = test_dataset.qas['test']
    test_qa_dict = test_dataset.qa_dict['test']

    if args.prompt_type == 'few-shot':
        few_shot_images = test_dataset.few_shot_images

        if isinstance(few_shot_images[0], list):
            few_shot_images = [img[0] for img in few_shot_images]

        if isinstance(few_shot_images[0], str):
            few_shot_images = [Image.open(img).convert('RGB') for img in few_shot_images]
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
        logging.info(f'Test Accuracy: {test_acc:.4f}')

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