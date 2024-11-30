import os
import os.path as osp
import pathlib
import numpy as np
import re
from dataclasses import dataclass
from typing import Dict, Sequence, List, Union
import json
import pandas as pd
import copy
from PIL import Image
from packaging import version
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s | %(message)s]',
    datefmt='%d-%b-%y %H:%M:%S'
)

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP
)

import transformers
import tokenizers
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from transformers import LlamaForCausalLM
from transformers import HfArgumentParser, EarlyStoppingCallback

import llava
from llava import conversation as conversation_lib
from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN
)
from llava.train.llava_trainer import LLaVATrainer
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from old_llava import LlavaLlamaForCausalLM as OldLlavaLlamaForCausalLM

DEFAULT_SEED = 42
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(
    tokenizers.__version__
) >= version.parse('0.14')

local_rank = None

@dataclass
class ModelArguments:
    model_name: str = 'llava-v1.5-7b'
    version: str = 'v1.5'
    path: str = 'liuhaotian/llava-v1.5-7b'
    vision_tower: str = 'openai/clip-vit-large-patch14'
    hf_cache_dir: str = '/data/hf_models'
    attn_implementation: str = 'flash_attention_2'
    start_stage: int = 0
    pretrain_mm_mlp_adapter: str = None
    mm_vision_select_layer: int = -1
    mm_projector_type: str = 'linear'
    mm_use_im_start_end: bool = False
    mm_use_im_patch_token: bool = True
    mm_patch_merge_type: str = 'flat'
    mm_vision_select_feature: str = 'patch'

@dataclass
class DataArguments:
    dataset_name: str = 'vqa-rad'
    image_folder: str = '/data/vqa-rad/images'
    data_path: str = '/data/vqa-rad/closed/train.jsonl'
    eval_data_path: str = None
    is_multimodal: bool = False
    image_aspect_ratio: str = 'pad'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Distributed computing config
    fsdp: Union[bool,str,List] = False
    train_stage: int = 1
    remove_unused_columns: bool = False # Must be fixed to False

    # Quantization config
    bits: int = 16
    double_quant: bool = True
    quant_type: str = 'nf4'

    # Compute dtype
    fp16: bool = False
    bf16: bool = False

    # Optimizer
    optim: str = 'adamw_torch'

    # Option to checkpoint gradients
    gradient_checkpointing: bool = True

    # LoRA config
    lora_enable: bool = True
    lora_r: int = 128
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ''
    lora_bias: str = 'none'
    use_rslora: bool = False

    # Projection layer config
    tune_mm_mlp_adapter: bool = False
    freeze_mm_mlp_adapter: bool = False
    mm_projector_lr: float = 2e-5

    # LLM backbone config
    freeze_backbone: bool = False

    # Tokenization config
    model_max_length: int = 512

    # Option to quasi-randomly sample examples with similar input lengths into a batch
    # Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L139
    group_by_modality_length: bool = False

    # Checkpoint directory
    output_dir: str = '../ckpts'

    # Deepspeed config
    deepspeed: str = '../configs_deepspeed/zero2.json'

    # Platform to log the results at
    report_to: Union[str, List[str]] = 'wandb'

    # Seed for training
    seed: int = DEFAULT_SEED

    # Model selection arguments
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'eval_loss'
    greater_is_better: bool = False

    # Option to force training even when previous checkpoint exists
    force_train: bool = False

def rank0_print(*args):
    '''Prints the contents to stdout only if the local rank is 0.'''

    if local_rank == 0:
        print(*args)

def write_jsonl(data, path):
    '''Saves a JSONL file given a list of dictionary records.'''

    with open(path, 'w') as fh:
        for item in data:
            json.dump(item, fh)
            fh.write('\n')

def read_jsonl(path):
    '''Reads a JSONL file and returns a list of dictionary records.'''

    with open(path, 'r') as fh:
        data = [json.loads(line) for line in fh]

    return data

def print_trainable_params(model, rank0=False):
    '''Prints the number of trainable parameters in the model.'''

    n_trainable_params = 0
    n_all_params = 0
    for _, param in model.named_parameters():
        n_all_params += param.numel()
        
        if param.requires_grad:
            n_trainable_params += param.numel()

    if rank0:
        rank0_print(
            f'Trainable: {n_trainable_params} || '
            f'All: {n_all_params} || '
            f'Trainable %: {100. * n_trainable_params / n_all_params:.2f}'
        )
    else:
        print(
            f'Trainable: {n_trainable_params} || '
            f'All: {n_all_params} || '
            f'Trainable %: {100. * n_trainable_params / n_all_params:.2f}'
        )

'''
    LoRA-related utility functions.

'''

def fetch_param(param, ignore_status=False, name=None):
    '''Retrieves the given parameter.'''

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
# NOTE: get_peft_model_state_dict returns the state dict of the PEFT model
def get_peft_state_lora(named_params, lora_bias):
    if lora_bias == 'none':
        to_return = {k: t for k, t in named_params if 'lora_' in k}
    
    elif lora_bias == 'all':
        to_return = {k: t for k, t in named_params if 'lora_' in k or 'bias' in k}
    
    elif lora_bias == 'lora_only':
        to_return = {}
        maybe_lora_bias = {} # Trying to fetch all LoRA bias layers
        lora_bias_names = set()
        
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            
            elif "bias" in k:
                maybe_lora_bias[k] = t
        
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    
    to_return = {k: fetch_param(v, ignore_status=True) for k, v in to_return.items()}
    
    return to_return

def get_peft_state_non_lora(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    
    to_return = {k: fetch_param(v, ignore_status=True).cpu() for k, v in to_return.items()}
    
    return to_return

def get_mm_adapter_state(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: fetch_param(v, ignore_status=True).cpu() for k, v in to_return.items()}
    
    return to_return

def find_all_linear_names(model):
    '''Returns all of the linear-layer module names in a LLaVA model to be adapted.'''

    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    
    for name, module in model.named_modules():
        # Skip the projector, vision_tower, and vision_resampler modules
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        # If linear layer
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)

'''
    Model saving utility functions.

'''

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    '''Collects the state dict and dump to disk.'''

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']

        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = osp.dirname(output_dir)

        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = osp.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, osp.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, osp.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

'''
    Utility functions for tokenization and dataset preprocessing.

'''

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel
):
    '''
        Resize the tokenizer and embeddings.
    
        NOTE: This is the unoptimized version that may make your embedding size not be divisible by 64.
    
    '''
    
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        # get_input_embeddings() -> |V| x d_model Embedding layer
        # get_output_embeddings() -> d_model x |V| shape linear layer weights for the LM head
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data # Returned with shape (|V|, d_model)

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        # Initialize the embeddings for the new tokens as the average of existing tokens
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    '''Tokenize a list of strings.'''
    
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0] # Skipping the system prompt
    tokenized_lens = tokenized_lens[1:] # Tokenized lengths for each message in conversation
    target[:cur_idx] = IGNORE_INDEX
    
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        # +2 corresponds to the '###' separator; i.e., loss is calculated for generating this separator.
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    '''
        Add speaker and start/end signal on each round.

        Example:
        Input: [{'user': 'user message', 'gpt': 'model message'}]
        Output: """
            System prompt

            ### USER: user message
            ### ASSISTANT: model message
            ###
        """
    '''
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    
    for sentence in source:
        from_str = sentence["from"]
        
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        
        else:
            from_str = 'unknown'
        
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    
    conversation += BEGIN_SIGNAL
    
    return conversation

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    '''
        Preprocesses a list of conversations with images.
        
        Converts to following format: <image>\n<text>.
        If 'mmtag' conversation style is used, converts to: <Image><image></Image>\n<text>.
        If 'mm_use_start_end' is True, converts to: <im_start><image><im_end>\n<text>.

    '''
    
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    # NOTE: sources = list of dictionaries
    # Each dictionary is either a user message or model response in a conversation
    for source in sources:
        for sentence in source:
            # If there is an image in the message
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                n_images = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence['value']))
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()

                if data_args.mm_use_im_patch_token:
                    if data_args.mm_use_im_start_end:
                        sentence['value'] += n_images * (
                            '\n' + DEFAULT_IM_START_TOKEN + \
                            DEFAULT_IMAGE_PATCH_TOKEN * data_args.image_token_len + \
                            DEFAULT_IM_END_TOKEN
                        )
                    else:
                        sentence['value'] += n_images * (
                            '\n' + DEFAULT_IMAGE_PATCH_TOKEN * data_args.image_token_len
                        )
                else:
                    if data_args.mm_use_im_start_end:
                        sentence['value'] = n_images * (
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                        ) + sentence['value']
                    else:
                        sentence['value'] = n_images * (DEFAULT_IMAGE_TOKEN + '\n') + sentence['value']

                sentence['value'] = sentence['value'].strip()

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>'
                    )
            
    return sources

def preprocess_llama_2(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(input_ids=input_ids, labels=targets)

def preprocess_v1(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}" # Checking if human and AI is going back and forth
            conv.append_message(role, sentence["value"])
        
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt'
        ) for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True, # Truncate to max length
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):
        # Total length of the target IDs excluding the pad tokens
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep # Add back the separator to the instruction

            if has_image:
                # Tokenize and insert IMAGE_TOKEN_INDEX in place of <image> token IDs
                # -2 to remove ' '
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            # Mask all tokens in the instruction, except for the bos_token
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(input_ids=input_ids, labels=targets)

def preprocess_mpt(sources, tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(input_ids=input_ids, labels=targets)

def preprocess_plain(sources: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess(sources: Sequence[str], tokenizer: PreTrainedTokenizer, has_image: bool = False) -> Dict:
    '''
        Given a list of sources, each is a conversation list. This transform:
        1. Add signal '### ' at the beginning each sentence, with end signal '\n';
        2. Concatenate conversations together;
        3. Tokenize the concatenated conversation;
        4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    
    '''

    conv_version = conversation_lib.default_conversation.version

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    
    # NOTE: This is the default
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image and 'v0' not in conv_version:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image and 'v0' not in conv_version:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

# More or less the same as the llava-med version
class SupervisedDataset(Dataset):
    '''Dataset for supervised fine-tuning.'''

    def __init__(
        self, 
        data_path: str, 
        tokenizer: PreTrainedTokenizer, 
        data_args: DataArguments, 
        list_data_dict: Dict = None
    ):
        super(SupervisedDataset, self).__init__()
        
        # This will be a list of dictionaries, as read from a JSON file
        if list_data_dict is None:
            data_ext = osp.splitext(data_path)[-1]
            if data_ext == '.jsonl':
                list_data_dict = read_jsonl(data_path)
            elif data_ext == '.json':
                list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Loading and preprocessing the dataset...")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        
        # Length of each sample = image token length (128) + conversation lengths
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        
        return length_list

    # Used for the group_by_modality_length option
    @property
    def modality_lengths(self):
        length_list = []
        
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        
        if isinstance(i, int):
            sources = [sources]

        assert len(sources) == 1
        
        # Check if the QAs are visual
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(osp.join(image_folder, image_file)).convert('RGB')
            
            # Pad image to square
            if self.data_args.image_aspect_ratio == 'pad':

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    
                    # If the image is already square
                    if width == height:
                        return pil_img
                    
                    # If width is longer, pad image to be (width, width)
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    
                    # If height is longer, pad image to be (height, height)
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                
                # Background padding color = mean of each image channel
                # processor.image_mean is a list of normalized pixel values of size 3
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            try:
                sources = preprocess_multimodal(
                    copy.deepcopy([e['conversations'] for e in sources]),
                    self.data_args
                )
            except:
                # NOTE: LLaVA-Med data has a typo
                sources = preprocess_multimodal(
                    copy.deepcopy([e['conversatons'] for e in sources]),
                    self.data_args
                )
        else:
            try:
                sources = copy.deepcopy([e['conversations'] for e in sources])
            except:
                sources = copy.deepcopy([e['conversatons'] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i])
        )
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    '''Collate examples for supervised fine-tuning.'''

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args) -> Dict:
    '''Make dataset and collator for supervised fine-tuning.'''
    
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=data_args.data_path, 
        data_args=data_args
    )

    if not data_args.eval_data_path is None and data_args.eval_data_path != '' and data_args.eval_data_path[-1] != '/':
        eval_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.eval_data_path,
            data_args=data_args
        )
    else:
        eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    local_rank = training_args.local_rank # NOTE: local_rank is inherited from TrainingArguments class

    # Set up wandb config
    report_to = training_args.report_to
    to_wandb = (
        (isinstance(report_to, str) and report_to == 'wandb') or \
        (isinstance(report_to, list) and 'wandb' in report_to)
    )
    if to_wandb:
        # e.g., llava-v0-7b_start-0,train-1, llava-med-7b_start-2,train-3_vqa-rad
        project_name = '_'.join(training_args.output_dir.split('/')[2:4])
        if 'train-3' in project_name:
            project_name += f'_{training_args.output_dir.split("/")[4]}'
        
        os.environ['WANDB_PROJECT'] = project_name

        # TODO: Check if this sets the name properly
        run_name = ''
        if training_args.lora_enable:
            if training_args.bits in [4,8]:
                run_name += 'qlora#'
            else:
                run_name += 'lora#'
            
            run_name += f'r={training_args.lora_r},'
            run_name += f'alpha={training_args.lora_alpha},'
            run_name += f'dropout={training_args.lora_dropout},'
            run_name += f'bias={training_args.lora_bias},'
            run_name += f'rslora={training_args.use_rslora}'
        else:
            run_name += 'ft_'
        
        run_name += 'train#'
        run_name += f'epochs={training_args.num_train_epochs},'
        run_name += f'batch_per_device={training_args.per_device_train_batch_size},'
        run_name += f'lr={training_args.learning_rate},'
        run_name += f'mm.lr={training_args.mm_projector_lr},'
        run_name += f'decay={training_args.weight_decay},'
        run_name += f'warmup={training_args.warmup_ratio},'
        run_name += f'scheduler={training_args.lr_scheduler_type},'
        run_name += f'tune.mm={training_args.tune_mm_mlp_adapter},'
        run_name += f'freeze.mm={training_args.freeze_mm_mlp_adapter},'
        run_name += f'freeze.backbone={training_args.freeze_backbone}'

        os.environ['WANDB_NAME'] = run_name

    # Set up precision
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # Set up quantization config and load the model checkpoint
    bnb_args = {}
    if training_args.bits in [4,8]:
        bnb_args.update(dict(
            #device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"], # Keep projector in 16 bits
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # Multimodal LLM
    if model_args.vision_tower is not None:
        # MPT-based LLaVA model
        if 'mpt' in model_args.model_name or 'mpt' in model_args.path:
            # Retrieve pretrained model config
            pretrained_config = AutoConfig.from_pretrained(
                model_args.path, cache_dir=model_args.hf_cache_dir, trust_remote_code=True
            )
            pretrained_config.attn_config['attn_impl'] = model_args.attn_implementation
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.path,
                config=pretrained_config,
                cache_dir=model_args.hf_cache_dir,
                **bnb_args
            )
        # Old-version LLaVA/LLaVA-Med model
        elif any([
            'llava-v0' in model_args.model_name or 'llava-v0' in model_args.path,
            'llava-med' in model_args.model_name or 'llava-med' in model_args.path
        ]):
            model = OldLlavaLlamaForCausalLM.from_pretrained(
                model_args.path,
                attn_implementation=model_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_args
            )
        # LLaMA/Vicuna-based LLaVA model
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.path,
                cache_dir=model_args.hf_cache_dir,
                attn_implementation=model_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None), # Weights data type
                **bnb_args
            )
    # Text-only LLM
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.path,
            cache_dir=model_args.hf_cache_dir,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_args
        )
    model.config.use_cache = False

    # Option to freeze the LLM backbone
    if training_args.freeze_backbone:
        model.model.requires_grad_(False)

    # Set up compute dtype when model is loaded in 4 or 8 bits
    if training_args.bits in [4,8]:
        model.config.torch_dtype=(
            torch.float32 if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        
        # NOTE: Sets up the model for training by:
        # - Casting layernorm to fp32 (for numerical stability; fp16 can easily result in overflow)
        # - Making the output embedding layer require gradients
        # - Casting the LM head to fp32
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    # Enable gradient checkpointing for better memory efficiency
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # Called when forward() is called on input embedding layers
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Option for LoRA
    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout, # Default is 0
            bias=training_args.lora_bias, # Default is None
            task_type='CAUSAL_LM',
            use_rslora=training_args.use_rslora # Option for rank-stabilized LoRA
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        
        rank0_print('Adding LoRA layers...')
        # Returns a new "PeftModelForCausalLM" model 
        # with only the LoRA parameters set as trainable (i.e., requires_grad=True).
        model = get_peft_model(model, lora_config)

    # Load pretrained tokenizer
    if 'mpt' in model_args.model_name or 'mpt' in model_args.path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.path,
            cache_dir=model_args.hf_cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side='right'
        )
    elif any([
        'llava-v0' in model_args.model_name or 'llava-v0' in model_args.path,
        'llava-med' in model_args.model_name or 'llava-med' in model_args.path
    ]):
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.path,
            model_max_length=training_args.model_max_length,
            padding_side='right',
            use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.path,
            cache_dir=model_args.hf_cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side='right',
            use_fast=False,
        )

    # Add padding token and set up conversation template
    if model_args.version == 'v0': # Vicuna v0
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token='[PAD]'),
                tokenizer=tokenizer,
                model=model,
            )

        conversation_lib.default_conversation = conversation_lib.conv_templates['llava_v0']

    elif model_args.version == 'v0.5': # Vicuna v0.5
        tokenizer.pad_token = tokenizer.unk_token
    else: # Vicuna >= v1.0 or other LLM
        tokenizer.pad_token = tokenizer.unk_token
    
        # Use the conversation template appropriate for the model version
        # Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py#L372
        # NOTE: This was originally within the else statement above
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates['vicuna_v1']

    # Set up CLIP ViT
    if model_args.vision_tower is not None:
        if model_args.version in ['v0', 'v0.5']:
            # NOTE: Additional layer due to model being a PeftModelForCausalLM object
            model_vision_dict = model.model.model.initialize_vision_modules(
                vision_tower=model_args.vision_tower,
                mm_vision_select_layer=model_args.mm_vision_select_layer,
                pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
                tune_mm_mlp_adapter=training_args.tune_mm_mlp_adapter,
                dtype=compute_dtype
            )

            # Vision tower is on CPU
            model.model.model.vision_tower[0].to(
                dtype=(torch.bfloat16 if training_args.bf16 else torch.float16),
                device=training_args.device
            )

            vision_config = model_vision_dict['vision_config']
            data_args.image_token_len = model_vision_dict['image_token_len']
            data_args.image_processor = model_vision_dict['image_processor']
            data_args.is_multimodal = True

            model.config.image_aspect_ratio = data_args.image_aspect_ratio
            model.config.tokenizer_padding_side = tokenizer.padding_side # Left/right
            model.config.tokenizer_model_max_length = tokenizer.model_max_length # Max sequence length

            # Tuning only the projector layer
            model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
            if training_args.tune_mm_mlp_adapter:
                model.requires_grad_(False)
                for p in model.model.model.mm_projector.parameters():
                    p.requires_grad = True

            # Option to freeze the projector layer
            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            for p in model.model.model.mm_projector.parameters():
                p.requires_grad = False if training_args.freeze_mm_mlp_adapter else True
            
            # Set up compute dtype for the projector layer
            if training_args.bits in [4,8]:
                model.model.model.mm_projector.to(dtype=compute_dtype, device=training_args.device)

            model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
            vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_projector_lr = training_args.mm_projector_lr # Learning rate for projector; NOTE: Not originally used in LLaVA-Med repo.
            model.config.mm_use_im_patch_token = data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token
            model.model.initialize_vision_tokenizer(
                mm_use_im_start_end=model_args.mm_use_im_start_end,
                tokenizer=tokenizer,
                device=training_args.device,
                tune_mm_mlp_adapter=training_args.tune_mm_mlp_adapter,
                pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
            )

            # TODO: Double-check what's going on here.
            params_no_grad = [n for n,p in model.named_parameters() if not p.requires_grad]
            if len(params_no_grad) > 0:
                if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                    if len(params_no_grad) < 10:
                        if training_args.local_rank == 0 or training_args.local_rank == -1:
                            logging.warning('Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
                                len(params_no_grad), params_no_grad
                            ))
                    else:
                        if training_args.local_rank == 0 or training_args.local_rank == -1:
                            logging.warning('Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                                len(params_no_grad), ', '.join(params_no_grad[:10])
                            ))
                    
                    if training_args.local_rank == 0 or training_args.local_rank == -1:
                        logging.warning('Attempting to use FSDP with partially frozen parameters, this is experimental.')
                        logging.warning(
                            'As of 4/30/23, this feature requires PyTorch-nighly build. '
                            'See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining'
                        )

                    def patch_FSDP_use_orig_params(func):
                        def wrap_func(*args, **kwargs):
                            use_orig_params = kwargs.pop('use_orig_params', True)
                            return func(*args, **kwargs, use_orig_params=use_orig_params)
                        return wrap_func
                    
                    FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)
        else:
            # Initializes the actual vision tower model and adds it to the model
            # NOTE: All of the vision towers are set to be not trainable. (model.base_model.model.model.vision_tower)
            # NOTE: All of the projection layers are set to be trainable. (model.base_model.model.model.mm_projector)
            model.get_model().initialize_vision_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
            
            vision_tower = model.get_vision_tower() # On CPU
            vision_tower.to(
                dtype=(torch.bfloat16 if training_args.bf16 else torch.float16), 
                device=training_args.device
            )
            data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True

            model.config.image_aspect_ratio = data_args.image_aspect_ratio
            model.config.tokenizer_padding_side = tokenizer.padding_side # Left/right
            model.config.tokenizer_model_max_length = tokenizer.model_max_length # Max sequence length

            # Tuning only the projector layer
            # NOTE: Before here, all weights except LoRA weights and projector weights are not trainable.
            model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
            if training_args.tune_mm_mlp_adapter:
                model.requires_grad_(False)
                
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True

            # Option to freeze the projector layer
            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False if training_args.freeze_mm_mlp_adapter else True

            # Set up compute dtype for the projector layer
            if training_args.bits in [4,8]:
                model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

            model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_projector_lr = training_args.mm_projector_lr # Learning rate for projector
            training_args.use_im_start_end = model_args.mm_use_im_start_end
            model.config.mm_use_im_patch_token = data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token
            model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4,8]:
        for name, module in model.named_modules():
            # Set LoRA layers to BF16 if applicable
            # NOTE: After get_peft_model(), the LoRA layer weights have dtype=torch.float32
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            
            # Set layernorm to 32 bits
            if 'norm' in name:
                module = module.to(torch.float32)

            # Set LM head and embeddings to 16 bits
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Load dataset
    if data_args.eval_data_path is None or data_args.eval_data_path == '' or data_args.eval_data_path[-1] == '/':
        data_ext = osp.splitext(data_args.data_path)[-1]
        if data_ext == '.jsonl':
            raw_data = read_jsonl(data_args.data_path)
        elif data_ext == '.json':
            raw_data = json.load(open(data_args.data_path))
        else:
            raise NotImplementedError(f'Unsupported data format: {data_ext}.')
        
        # Take random 80-20 split
        train_idxs, eval_idxs = train_test_split(np.arange(0,len(raw_data)), test_size=0.2, random_state=training_args.seed)
        train_data = [raw_data[i] for i in train_idxs]
        eval_data = [raw_data[i] for i in eval_idxs]

        train_dataset = SupervisedDataset(
            data_path='', tokenizer=tokenizer, data_args=data_args, list_data_dict=train_data
        )
        eval_dataset = SupervisedDataset(
            data_path='', tokenizer=tokenizer, data_args=data_args, list_data_dict=eval_data
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        data_module = dict(
            train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
        )
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Print proportion of parameters being trained
    print_trainable_params(model, rank0=True)

    trainer = LLaVATrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=1,
            early_stopping_threshold=0. # Absolute difference between val. metric and best val. metric
        )],
        **data_module
    )

    if not training_args.force_train and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # NOTE: Trainer state keeps track of e.g., the epoch that training is at, the best loss/metric obtained so far.
    # Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L35
    trainer.save_state() # Saved to training_args.output_dir

    # Save loss history from training
    trainer_history = pd.DataFrame(trainer.state.log_history)
    trainer_history.to_csv(osp.join(training_args.output_dir, 'trainer_history.csv'), index=False)

    model.config.use_cache = True

    if training_args.lora_enable:
        # Retrieve the state dictionaries, collecting across the distributed machines
        state_dict = get_peft_state_lora(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora(model.named_parameters())
        
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir) # Save model configs
            model.save_pretrained(training_args.output_dir, state_dict=state_dict) # Save model weights
            torch.save(non_lora_state_dict, osp.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
if __name__ == '__main__':
    train()