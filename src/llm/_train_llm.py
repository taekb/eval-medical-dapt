import os
import os.path as osp
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import random
import pandas as pd
import numpy as np
import pathlib
from functools import partial
from dataclasses import dataclass
from typing import Union, List
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

import huggingface_hub

import torch

import bitsandbytes as bnb
import transformers
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER
from transformers import Trainer, HfArgumentParser, EarlyStoppingCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from peft import LoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training

import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils import LLAMA3_CHAT_TEMPLATE, MISTRAL_CHAT_TEMPLATE

from dataset import (
    CASISenseDataset,
    MIMICIIISenseDataset,
    MedNLIDataset,
    EHRNoteQADataset,
    n2c2Dataset,
    MedQADataset,
    MedMCQADataset,
    PubMedQADataset,
    qa_collate_fn
)

DEFAULT_SEED = 42
MODELS_REQ_LOGIN = [
    'wanglab/ClinicalCamel-70B',
    'meta-llama/Meta-Llama-3-70B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-70b-hf',
    'epfl-llm/meditron-7b',
    'epfl-llm/meditron-70b',
    'mistralai/Mistral-7B-Instruct-v0.1'
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
    'pubmedqa': PubMedQADataset
}
IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    model_name: str = 'llama-2-7b'
    path: str = 'meta-llama/Llama-2-7b-hf'
    hf_cache_dir: str = '/data/hf_models'
    hf_api_key: str = ''
    attn_implementation: str = 'flash_attention_2'
    start_stage: int = 0

@dataclass
class DataArguments:
    dataset_name: str = 'medqa'
    data_path: str = 'bigbio/med_qa'
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Quantization config
    bits: int = 16
    double_quant: bool = True
    quant_type: str = 'nf4'

    # Compute dtype
    fp16: bool = False
    bf16: bool = True

    # Optimizer
    optim: str = 'adamw_torch'

    # Train configs
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    evaluation_strategy: str = 'epoch'
    eval_accumulation_steps: int = 1
    save_strategy: str = 'epoch'
    save_total_limit: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = 'cosine'
    logging_strategy: str = 'epoch'
    tf32: bool = True
    dataloader_num_workers: int = 4

    # LoRA config
    lora_enable: bool = True
    lora_r: int = 128
    lora_alpha: int = 16
    lora_dropout: float = 0.
    lora_weight_path: str = ''
    lora_bias: str = 'none'
    use_rslora: bool = False

    # Tokenization config
    model_max_length: int = 2048

    # Checkpoint directory
    output_dir: str = '../ckpts'

    # Platform to log the results at
    report_to: Union[str, List[str]] = 'wandb'

    # Seed for training
    seed: int = DEFAULT_SEED

    # Training stage
    train_stage: int = 3

    # Model selection arguments
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'eval_loss'
    greater_is_better: bool = False

    # Option to force training even when previous checkpoint exists
    force_train: bool = False

def random_seed(seed=DEFAULT_SEED, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def find_all_linear_names(args, model):
    '''Finds all linear modules to apply LoRA to.'''

    if args.bits == 4:
        cls = bnb.nn.Linear4bit
    elif args.bits == 8:
        cls = bnb.nn.Linear8bitLt
    else:
        cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

def train():
    '''Main training function.'''
    
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    local_rank = training_args.local_rank

    # Log in to HF if required
    if model_args.path in MODELS_REQ_LOGIN:
        huggingface_hub.login(token=model_args.hf_api_key)

    # Set up W&B
    report_to = training_args.report_to
    to_wandb = (
        (isinstance(report_to, str) and report_to == 'wandb') or \
        (isinstance(report_to, list) and 'wandb' in report_to)
    )
    if to_wandb: 
        os.environ['WANDB_PROJECT'] = f'{model_args.model_name}_{data_args.dataset_name}'

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
            run_name += '_'
        else:
            run_name += 'ft_'
        
        run_name += 'train#'
        run_name += f'epochs={training_args.num_train_epochs},'
        run_name += f'batch_per_device={training_args.per_device_train_batch_size},'
        run_name += f'lr={training_args.learning_rate},'
        run_name += f'decay={training_args.weight_decay},'
        run_name += f'warmup={training_args.warmup_ratio},'
        run_name += f'scheduler={training_args.lr_scheduler_type}'

        os.environ['WANDB_NAME'] = run_name

    # Set up precision
    if training_args.fp16:
        compute_dtype = torch.float16
    elif training_args.bf16:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # Set up quantization
    if training_args.bits in [4,8]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(training_args.bits == 4),
            load_in_8bit=(training_args.bits == 8),
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type
        )
    else:
        quantization_config = None

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.path,
        cache_dir=model_args.hf_cache_dir,
        quantization_config=quantization_config,
        attn_implementation='flash_attention_2',
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None)
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.path,
        cache_dir=model_args.hf_cache_dir,
        padding_side='right', # Fixed to right-side padding for all models
        use_fast=True
    )
    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if getattr(tokenizer, 'model_max_length', None) is None or tokenizer.model_max_length == VERY_LARGE_INTEGER:
        tokenizer.model_max_length = model.config.max_position_embeddings
    
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logging.warning('Mismatch between vocab size and embedding matrix.')
        model.resize_token_embeddings(len(tokenizer))

    # Add special tokens for MediTron
    if 'meditron' in model_args.model_name:
        new_tokens = ['<|im_start|>', '<|im_end|>']
        tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        model.resize_token_embeddings(len(tokenizer))
        modules_to_save = ['embed_tokens']
    else:
        modules_to_save = None

    # OpenBioLLM: Add Llama-3 tokenizer chat template
    if 'openbiollm' in model_args.model_name and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    # Mistral: Adjust chat template
    elif 'mistral-7b-v0.1' in model_args.model_name:
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    # Prepare model for mixed precision training if using QLoRA
    if training_args.bits in [4,8]: 
        model.config.torch_dtype = (
            torch.float16 if training_args.fp16 # FIXME: Not sure why we need this, actually
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    # Gradient checkpointing
    # NOTE: This must be called before calling get_peft_model().
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Add LoRA layers
    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(training_args, model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type='CAUSAL_LM',
            use_rslora=training_args.use_rslora,
            modules_to_save=modules_to_save
        )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            elif training_args.fp16:
                model.to(torch.float16)

        logging.info('Adding LoRA layers...')
        model = get_peft_model(model, lora_config)

        if local_rank == 0:
            model.print_trainable_parameters()

            if 'meditron' in model_args.model_name:
                logging.info('Note: For MediTron, the input embedding layer is set to be trainable.')

    # Add gradient masking for MediTron
    if 'meditron' in model_args.model_name:
        embeds = model.get_input_embeddings().weight
        new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        mask = torch.zeros(embeds.size(0), device='cuda')
        mask[new_token_ids] = 1
        mask = mask.unsqueeze(1)
        mask = mask.to(model.dtype)

        def apply_grad_mask(grad):
            return grad * mask.to(grad.dtype)
        
        embeds.register_hook(apply_grad_mask)

    # Initialize data loaders
    dataset_cls = DATASET_NAME_TO_CLS[data_args.dataset_name]
    train_dataset = dataset_cls(
        name=data_args.dataset_name,
        qa_dir=data_args.data_path,
        splits=['train'], 
        main_split='train', 
        seed=training_args.seed,
        hf_cache_dir=model_args.hf_cache_dir
    )
    train_dataset.load_and_apply_prompt_template(
        model_name=model_args.model_name,
        prompt_type='zero-shot-ft',
        tokenize=True,
        tokenizer=tokenizer
    )
    eval_dataset = dataset_cls(
        name=data_args.dataset_name,
        qa_dir=data_args.data_path,
        splits=['val'], 
        main_split='val', 
        seed=training_args.seed,
        hf_cache_dir=model_args.hf_cache_dir
    )
    eval_dataset.load_and_apply_prompt_template(
        model_name=model_args.model_name,
        prompt_type='zero-shot-ft',
        tokenize=True,
        tokenizer=tokenizer
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=partial(
            qa_collate_fn, 
            pad_token_id=tokenizer.pad_token_id, 
            return_dict=True,
            pad_left=False
        ),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=1,
            early_stopping_threshold=0.
        )]
    )

    if not training_args.force_train and list(pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
        if local_rank == 0 or local_rank == -1:
            logging.info('Checkpoint found. Resuming training from checkpoint...')
    else:
        trainer.train()

    trainer.save_state()

    # Save loss history
    trainer_history = pd.DataFrame(trainer.state.log_history)
    trainer_history_path = osp.join(training_args.output_dir, 'trainer_history.csv')
    trainer_history.to_csv(trainer_history_path, index=False)
    
    if local_rank == 0 or local_rank == -1:
        logging.info(f'Trainer history saved in: "{trainer_history_path}"')

    model.config.use_cache = True

    # Save model
    trainer.model.save_pretrained(training_args.output_dir)
    trainer.tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    train()