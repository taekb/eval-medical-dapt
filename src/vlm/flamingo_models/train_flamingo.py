import os
import os.path as osp
import sys
import argparse
import random
from dataclasses import dataclass
import pandas as pd
from functools import partial
from datetime import timedelta
from collections.abc import Iterable
from PIL import Image
import numpy as np
import json
import wandb
from einops import rearrange
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s | %(message)s]',
    datefmt='%d-%b-%y %H:%M:%S'
)

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import(
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group

from transformers import BitsAndBytesConfig
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

import open_clip
import open_flamingo
from open_flamingo.train.distributed import world_info_from_env
from open_flamingo.train.train_utils import (
    get_mp_policy_dtype, 
    get_cast_dtype,
    get_autocast,
    filter_state_dict_to_trainable
)

from infer_flamingo import create_model_and_transforms_with_quant

DEFAULT_SEED = 42
IGNORE_INDEX = -100

def random_seed(seed=DEFAULT_SEED, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

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

def get_config_str(args):
    '''Returns a string describing the training config.'''

    # TODO: Add LoRA option later
    config_str = 'ft_'
    config_str += 'train#'
    config_str += f'epochs={args.num_epochs},'
    config_str += f'batch={args.batch_size},'
    config_str += f'lr={args.learning_rate},'
    config_str += f'xattn={args.cross_attn_every_n_layers},'
    config_str += f'decay={args.weight_decay},'
    config_str += f'warmup={args.warmup_ratio},'
    config_str += f'scheduler={args.lr_scheduler},'
    config_str += f'freeze.embed={args.freeze_lm_embeddings}'

    return config_str

# Adapted from: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/distributed.py#L40
def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False

# Adapted from: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/distributed.py#L73
def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
                timeout=timedelta(minutes=90) # NOTE: Default NCCL timeout is 30min
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend, 
                init_method=args.dist_url,
                timeout=timedelta(minutes=90) # NOTE: Default NCCL timeout is 30min
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True
    else:
        # needed to run on single gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=1,
            rank=0,
            timeout=timedelta(minutes=90) # NOTE: Default NCCL timeout is 1800s = 30min
        )

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    device = torch.device(device)
    return device

def preprocess_images(images, image_processor):
    '''Preprocesses a list of images using the given CLIP image processor.'''

    image_tensors = [image_processor(s).unsqueeze(0) for s in images]
    image_tensors = torch.cat(image_tensors, dim=0)
    
    return image_tensors

@dataclass
class EarlyStopper:
    '''Early stopping class.'''

    patience: int = 1
    min_delta: float = 0.
    counter: int = 0
    min_val_loss: float = float('inf')
    should_stop: bool = False
    rank: int = None

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1

            if self.rank is not None and (self.counter < self.patience):
                if self.rank == 0:
                    logging.info(f'[EarlyStopper] Patience remaining: {self.patience - self.counter}')
            else:
                logging.info(f'[EarlyStopper] Patience remaining: {self.patience - self.counter}')
            
            if self.counter >= self.patience:
                return True
            
        return False

class FlamingoVQADataset(torch.utils.data.Dataset):
    '''VQA dataset class for training and evaluating Open-Flamingo models.'''

    def __init__(self, data, image_folder, image_processor, tokenizer, rank0=False):
        if not isinstance(data, list):
            raise RuntimeError('"data" is expected to be a list of JSON objects.')

        self.image_folder = image_folder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.rank0 = rank0

        self.load(data)

    # Load the raw data
    def load(self, data):
        image_tensor_list = []
        input_id_list = []
        label_id_list = []

        pbar = tqdm(data, desc='Preprocessing VQA pairs', unit='VQA pair', disable=(not self.rank0))
        for sample in pbar:
            image_name = sample['image']
            image_path = osp.join(self.image_folder, image_name)
            image = Image.open(image_path).convert('RGB')
            images = [image]
            image_tensor = preprocess_images(images, self.image_processor)
            image_tensor_list.append(image_tensor)

            try:
                conv = sample['conversations']
            except:
                conv = sample['conversatons']

            assert(conv[0]['from'] == 'human' and conv[1]['from'] == 'gpt')
            question = conv[0]['value']
            answer = conv[1]['value']

            question = question.replace('<image>\n', '<image>')
            assert('<image>' in question)
            
            # NOTE: Excluding eos_token in the count
            context_len = len(self.tokenizer(f'{question} ', return_tensors='pt').input_ids[0]) - 1
            
            qa_pair = f'{question} {answer.capitalize()}.<|endofchunk|>{self.tokenizer.eos_token}'
            tokenized_text = self.tokenizer(qa_pair, return_tensors='pt')
            input_ids = tokenized_text.input_ids
            label_ids = input_ids.clone()
            label_ids[0,:context_len] = IGNORE_INDEX # Mask out the context
            
            input_id_list.append(input_ids)
            label_id_list.append(label_ids)
            
        self.image_tensor_list = image_tensor_list
        self.input_id_list = input_id_list
        self.label_id_list = label_id_list
        
    def __len__(self):
        return len(self.input_id_list)

    def __getitem__(self, idx):
        if type(idx) == int:
            return (
                self.image_tensor_list[idx], 
                self.input_id_list[idx], 
                self.label_id_list[idx]
            )

        elif isinstance(idx, Iterable):
            return [(
                self.image_tensor_list[i], 
                self.input_id_list[i], 
                self.label_id_list[i]
            ) for i in idx]

        elif isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            step = idx.step if idx.step is not None else 1
            idx = range(start, stop, step)
            
            return [(
                self.image_tensor_list[i], 
                self.input_id_list[i], 
                self.label_id_list[i]
            ) for i in idx]

def collate_fn(qa_samples, tokenizer):
    '''Collate function for batching.'''

    image_tensor_list, input_id_list, label_id_list = zip(*qa_samples)
    
    # Stack all images together
    batch_image_tensor = torch.cat(image_tensor_list, dim=0)
    
    # Stack the input_ids and label_ids together
    batch_input_ids = pad_sequence(
        [ids[0] for ids in input_id_list],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    batch_label_ids = pad_sequence(
        [ids[0] for ids in label_id_list],
        batch_first=True,
        padding_value=IGNORE_INDEX
    )

    # Get the attention masks
    batch_attention_masks = batch_input_ids.ne(tokenizer.pad_token_id)

    return (batch_image_tensor, batch_input_ids, batch_label_ids, batch_attention_masks)

def print_trainable_params(model, rank, rank0=False):
    '''Prints the number of trainable parameters in the model.'''

    n_trainable_params = 0
    n_all_params = 0
    for _, param in model.named_parameters():
        n_all_params += param.numel()
        
        if param.requires_grad:
            n_trainable_params += param.numel()

    if rank0: 
        if rank == 0:
            print(
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

def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, save_full_state=False, params_to_save=None):
    '''
        Save training checkpoint with model, optimizer, and lr_scheduler state.
    
    '''

    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        os.makedirs(args.output_dir, exist_ok=True)

        # Save full training state
        if save_full_state:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optim_state,
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
        else:
            if params_to_save is not None:
                checkpoint_dict = {k: v for k,v in model_state.items() if k in params_to_save}
            else:
                checkpoint_dict = model_state
        
        ckpt_path = osp.join(args.output_dir, f'checkpoint_{epoch}.pt')

        torch.save(checkpoint_dict, ckpt_path)
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(ckpt_path, base_path=args.output_dir)

        logging.info(f'Checkpoint saved to "{ckpt_path}".')

        if args.delete_previous_checkpoint:
            if epoch > 0:
                for prev_epoch in range(0,epoch):
                    prev_ckpt_path = osp.join(args.output_dir, f'checkpoint_{prev_epoch}.pt')
                    if osp.exists(prev_ckpt_path):
                        os.remove(prev_ckpt_path)
                        logging.info(f'Removed "{prev_ckpt_path}".')

def train_one_epoch(
    args,
    model,
    epoch,
    train_loader,
    train_sampler,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id
):
    '''Training function for a single epoch.'''

    autocast = get_autocast(args.precision, cache_enabled=(not args.fsdp))
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer('<image>', add_special_tokens=False)['input_ids'][-1]
    endofchunk_token_id = tokenizer('<|endofchunk|>', add_special_tokens=False)['input_ids'][-1]
    
    model.train()
    train_sampler.set_epoch(epoch)

    total_loss = 0.
    pbar = tqdm(enumerate(train_loader), disable=(args.rank != 0), total=len(train_loader))
    for i, (image_tensor, input_ids, label_ids, attn_masks) in pbar:
        pbar.set_description(f'[Rank {args.rank}] Epoch {epoch+1}')

        image_tensor = image_tensor.to(device_id, dtype=cast_dtype, non_blocking=True)
        image_tensor = rearrange(image_tensor, '(b t f) c h w -> b t f c h w', t=1, f=1)
        input_ids = input_ids.to(device_id, dtype=cast_dtype, non_blocking=True)
        label_ids = label_ids.to(device_id, dtype=cast_dtype, non_blocking=True)
        attn_masks = attn_masks.to(device_id, dtype=cast_dtype, non_blocking=True)

        with autocast():
            loss = model(
                vision_x=image_tensor,
                lang_x=input_ids,
                attention_mask=attn_masks,
                labels=label_ids
            )[0]

        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()
        total_loss += divided_loss.detach().float() # Tensor

        if (not args.freeze_lm_embeddings) and (not args.fsdp or args.fsdp_use_orig_params):
            # Mask gradients for input embeddings s.t. we only update the <image> and <|endofchunk|> tokens
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = model.module.lang_encoder.get_input_embeddings().weight.grad
            
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(zero_mask[endofchunk_token_id])

            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = embed_grad * zero_mask
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = embed_grad * zero_mask

        # Step optimizer and log
        if (((i+1) % args.gradient_accumulation_steps) == 0) or (i == len(train_loader)-1):
            # Gradient clipping
            if args.fsdp:
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

            if args.rank == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}')

    pbar.close()
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    train_epoch_loss = total_loss / len(train_loader)

    return train_epoch_loss.item()

def eval_one_epoch(
    args,
    model,
    epoch,
    eval_loader,
    eval_sampler,
    device_id
):
    '''Evaluation function for a single epoch.'''

    autocast = get_autocast(args.precision, cache_enabled=(not args.fsdp))
    cast_dtype = get_cast_dtype(args.precision)
    model.eval()
    eval_sampler.set_epoch(epoch)

    total_loss = 0.
    pbar = tqdm(enumerate(eval_loader), disable=(args.rank != 0), total=len(eval_loader))
    for i, (image_tensor, input_ids, label_ids, attn_masks) in pbar:
        pbar.set_description(f'[Rank {args.rank}] Validation')
        #global_step = i + epoch * len(eval_loader)

        image_tensor = image_tensor.to(device_id, dtype=cast_dtype, non_blocking=True)
        image_tensor = rearrange(image_tensor, '(b t f) c h w -> b t f c h w', t=1, f=1)
        input_ids = input_ids.to(device_id, dtype=cast_dtype, non_blocking=True)
        label_ids = label_ids.to(device_id, dtype=cast_dtype, non_blocking=True)
        attn_masks = attn_masks.to(device_id, dtype=cast_dtype, non_blocking=True)

        with autocast(), torch.no_grad():
            loss = model(
                vision_x=image_tensor,
                lang_x=input_ids,
                attention_mask=attn_masks,
                labels=label_ids
            )[0]
            total_loss += loss.detach().float() # Tensor

        if args.rank == 0:
            pbar.set_postfix(eval_loss=f'{loss.item():.4f}')

    pbar.close()
    torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
    eval_epoch_loss = total_loss / len(eval_loader)

    return eval_epoch_loss.item()

# Adapted from: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/train.py
def train(args):
    '''Training function for Open-Flamingo models.'''
    
    args.run_name = f'{args.model_name}_{args.dataset_name}'

    # NOTE: 
    # Reference: https://discuss.pytorch.org/t/what-is-the-difference-between-rank-and-local-rank/61940/2
    # local_rank = Rank of each process within each node
    # global_rank = Rank of each process across all nodes
    # world_size = Total number of processes running in parallel
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError('"save_checkpoints_to_wandb" requires "report_to_wandb"')

    if args.fsdp and args.rank == 0:
        logging.info('Using FSDP.')
        if not args.fsdp_use_orig_params:
            logging.warning(
                "FSDP is running without fsdp_use_orig_params flag. "
                + "This is not recommended because it means we will use uniform weight decay"
                + " and train all embeddings, not just the newly added ones. "
                + "Note: OPT models are not compatible with fsdp_use_orig_params flag."
            )
        
        if args.fsdp_sharding_strategy == 'hybrid':
            logging.warning(
                "As of torch=2.0.1, the FSDP logic for optim_state_dict() is broken for hybrid sharding."
                + "To make this method work, we need to modify torch.distributed.fsdp._optim_utils.py"
                + "Copy and paste the code from the _optim_utils.py in this repo into the torch file."
                + "The main issue was the missing group kwarg on line 1596 in _all_gather_optim_state."
            )

    if args.dataset_name not in args.data_path:
        raise ValueError('Mismatch: "dataset_name" not in "data_path".')

    # Run locally
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Sets the device cuda index according to local_rank
    # NOTE: device = f'cuda:{args.local_rank}' (e.g., if 2 GPUs on 2 nodes, local_rank will be ((0,1),(0,1))
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize W&B logging
    if args.rank == 0:
        logging.info(f'Starting distributed training on {args.world_size} GPUs...')

        if args.report_to_wandb:
            wandb.login()
            wandb.init(
                project=f'{args.model_name}_{args.dataset_name}',
                name=get_config_str(args),
                config=vars(args)
            )

    # Set up quantization and dtype
    compute_dtype = get_mp_policy_dtype(args.precision)
    if args.bits in [4,8]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(args.bits == 4),
            load_in_8bit=(args.bits == 8),
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type
        )
    else:
        quantization_config = None

    # Initialize model on rank 0 and broadcast to other processes
    model, image_processor, tokenizer = create_model_and_transforms_with_quant(
        args.clip_vision_encoder_path,
        args.clip_vision_encoder_pretrained,
        args.lang_encoder_path,
        args.tokenizer_path if args.tokenizer_path else args.lang_encoder_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        #use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        quantization_config=quantization_config,
        use_flash_attn=(args.attn_implementation == 'flash_attention_2')
    )

    if 'med-flamingo' in args.model_name:
        ckpt_path = osp.join(
            args.hf_cache_dir, 
            'models--{}--{}'.format(*args.path.split('/')),
            'snapshots/7243cd83bd426ceade9c4de9844cc5e5f3ff75e0/model.pt' # Commit hash
        )
    elif 'open-flamingo' in args.model_name:
        ckpt_path = osp.join(
            args.hf_cache_dir, 
            'models--{}--{}'.format(*args.path.split('/')),
            'snapshots/68d9e9ae82043cf0524e3ff0c806fdab8cc6a8a6/checkpoint.pt' # Commit hash
        )
    else:
        ckpt_path = ''

    if not osp.exists(ckpt_path):
        if args.rank == 0:
            logging.warning(f'Manually constructed checkpoint path not found: "{ckpt_path}"')
        
        # NOTE: For FSDP, this can lead to lock file contention issues across processes
        ckpt_path = hf_hub_download(repo_id=args.path, filename='model.pt', cache_dir=args.hf_cache_dir)

    state_dict = torch.load(ckpt_path, map_location='cpu')
    params_to_save = list(state_dict.keys()) # List of trainable parameter names to checkpoint
    model.load_state_dict(state_dict, strict=False)
    print_trainable_params(model, args.rank, rank0=True)
    
    tokenizer.model_max_length = args.model_max_length
    tokenizer.padding_side = 'right'

    random_seed(args.seed, args.rank)
    
    # Initialize FSDP/DDP and ensure model is on GPU
    if args.fsdp:
        # Shows the total parameter count before sharding; model still on CPU.
        logging.info(
            f'Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}'
        )

        if args.precision != 'fp32':
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=compute_dtype, # cross-GPU reduce and gather dtype
                buffer_dtype=compute_dtype
            )
        else:
            mp_policy = None

        if args.fsdp_sharding_strategy == 'hybrid':
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(_get_default_group())
            args.my_group = intra_node_group # For optimizer saving
            process_group = (intra_node_group, inter_node_group) # For FSDP init
        else:
            args.my_group = None
            process_group = None

        wrapper_kwargs = dict(
            process_group=process_group, # Group of processes over which sharding occurs; used for FSDP all-gather and reduce-scatter comms.
            cpu_offload=CPUOffload(offload_params=False), # Specifies whether to offload parameters to CPU when not in use
            device_id=device_id,
            sync_module_states=True, # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=( # Reference: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy
                ShardingStrategy.FULL_SHARD 
                if args.fsdp_sharding_strategy == 'full'
                else ShardingStrategy.HYBRID_SHARD # This limits all-reduce & reduce-scatter comms to be "within" nodes.
            ),
            use_orig_params=args.fsdp_use_orig_params,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True
        )
        model.wrap_fsdp(wrapper_kwargs, device_id)
        ddp_model = model
    
        logging.info(f'After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}')
        logging.info(f'After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}')
    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing (disabled by default)
    if args.gradient_checkpointing:
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, '_use_gradient_checkpointing', False)
            and not isinstance(m, FSDP)
            and not isinstance(m, CheckpointWrapper)
        )

    # Initialize optimizer
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], 'exclude_from_optimizer', False),
            params_to_optimize
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        # Apply weight decay only to params in x-attn layers
        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if 'gated_cross_attn' in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)

            return [
                {"params": params_with_wd, "weight_decay": args.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0}
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize), lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    # Initialize data loaders
    dataset_args = dict(
        image_folder=args.image_folder, 
        image_processor=image_processor, 
        tokenizer=tokenizer,
        rank0=(args.rank == 0)
    )
    if args.eval_data_path is None or args.eval_data_path == '' or args.eval_data_path[-1] == '/':
        data_ext = osp.splitext(args.data_path)[-1]
        if data_ext == '.jsonl':
            raw_data = read_jsonl(args.data_path)
        elif data_ext == '.json':
            raw_data = json.load(open(args.data_path))
        else:
            raise NotImplementedError(f'Unsupported data format: {data_ext}.')

        # Take random 80-20 split
        train_idxs, eval_idxs = train_test_split(np.arange(0,len(raw_data)), test_size=0.2, random_state=args.seed)
        train_data = [raw_data[i] for i in train_idxs]
        eval_data = [raw_data[i] for i in eval_idxs]
    else:
        data_ext = osp.splitext(args.data_path)[-1]
        if data_ext == '.jsonl':
            train_data = read_jsonl(args.data_path)
            eval_data = read_jsonl(args.eval_data_path)
        elif data_ext == '.json':
            train_data = json.load(open(args.data_path))
            eval_data = json.load(open(args.eval_data_path))
        else:
            raise NotImplementedError(f'Unsupported data format: {data_ext}.')

    train_dataset = FlamingoVQADataset(train_data, **dataset_args)
    eval_dataset = FlamingoVQADataset(eval_data, **dataset_args)
    train_sampler = DistributedSampler(train_dataset, rank=args.rank, num_replicas=args.world_size, shuffle=True, seed=args.seed)
    eval_sampler = DistributedSampler(eval_dataset, rank=args.rank, num_replicas=args.world_size)

    # Reference: https://pytorch.org/docs/stable/notes/randomness.html
    dataloader_args = dict(
        collate_fn=partial(collate_fn, tokenizer=tokenizer), 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        persistent_workers=True,
        pin_memory=True
    )
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **dataloader_args)
    eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, **dataloader_args)
    total_training_steps = args.num_epochs * len(train_loader)
    num_warmup_steps = int(args.warmup_ratio * total_training_steps)

    # Initialize lr scheduler
    if args.lr_scheduler == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )

    ddp_model.train()
    early_stopper = EarlyStopper(patience=1, min_delta=0)
    loss_history = dict(train=[], val=[])
    best_epoch = None
    best_eval_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_epoch_loss = train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            train_sampler=train_sampler,
            device_id=device_id
        )
        loss_history['train'].append(train_epoch_loss)

        eval_epoch_loss = eval_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            eval_loader=eval_loader,
            eval_sampler=eval_sampler,
            device_id=device_id
        )
        loss_history['val'].append(eval_epoch_loss)

        if args.rank == 0 and args.report_to_wandb:
            wandb.log(dict(train_loss=train_epoch_loss, eval_loss=eval_epoch_loss), commit=True)

        # Save checkpoint
        torch.distributed.barrier()
        if eval_epoch_loss < best_eval_loss:
            if args.rank == 0:
                logging.info(f'New best model: {eval_epoch_loss:.4f} < {best_eval_loss:.4f}. Saving checkpoint...')
            
            best_epoch = epoch
            best_eval_loss = eval_epoch_loss

            save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args, params_to_save=params_to_save)
        
        torch.distributed.barrier()

        # Check for early stopping
        if early_stopper.early_stop(eval_epoch_loss):
            if args.rank == 0:
                logging.info('Early stopping triggered. Ending training.')
            
            torch.distributed.barrier()
            break

    if args.rank == 0:
        logging.info('Training finished.')

    # Create symlink to final model checkpoint
    if args.rank == 0:
        ckpt_path = osp.join(args.output_dir, f'checkpoint_{best_epoch}.pt')
        link_path = osp.join(args.output_dir, 'checkpoint.pt')

        try:
            if osp.exists(link_path):
                os.remove(link_path)
            
            os.symlink(ckpt_path, link_path)
            logging.info(f'Updated symlink to best checkpoint: "{link_path}".')
        except:
            logging.error('Failed to create symlink for final model checkpoint.')

    # Save loss history
    if args.rank == 0:
        loss_history_path = osp.join(args.output_dir, 'loss_history.csv')
        loss_history_df = pd.DataFrame(loss_history)
        loss_history_df.to_csv(loss_history_path, index=False, header=True)
        logging.info(f'Loss history saved in: "{loss_history_path}"')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument('--model_name', default='open-flamingo-9b', type=str)
    parser.add_argument(
        '--path', 
        default='openflamingo/OpenFlamingo-9B-deprecated', 
        type=str,
        help='Path to model checkpoint.'
    )
    parser.add_argument('--clip_vision_encoder_path', default='ViT-L-14', type=str)
    parser.add_argument('--clip_vision_encoder_pretrained', default='openai', type=str)
    parser.add_argument('--lang_encoder_path', default='/data/llama-v1-weights/7B', type=str)
    parser.add_argument('--tokenizer_path', default='/data/llama-v1-weights/7B', type=str)
    parser.add_argument('--model_max_length', default=2048, type=int)
    parser.add_argument('--cross_attn_every_n_layers', default=4, type=int)
    parser.add_argument('--attn_implementation', default='flash_attention_2', type=str)
    parser.add_argument('--hf_cache_dir', default=None, type=str)
    
    # Training config
    parser.add_argument('--resume_from_checkpoint', default=None, type=str)
    parser.add_argument('--delete_previous_checkpoint', default=False, action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str) # Options: cosine, constant, linear
    parser.add_argument('--warmup_ratio', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true')
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--offline', default=False, action='store_true')
    parser.add_argument(
        '--freeze_lm_embeddings', 
        default=False, 
        action='store_true',
        help=(
            'Freeze the LLM token embeddings during training. '
            'Otherwise, train the <image> and <|endofchunk|> embeddings.'
        )
    )
    parser.add_argument('--dataset_name', default='vqa-rad', type=str)
    parser.add_argument('--image_folder', default='/data/vqa-rad/images', type=str)
    parser.add_argument('--data_path', default='/data/vqa-rad/combined/train.jsonl', type=str)
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument(
        '--output_dir', 
        default='./', 
        type=str,
        help='Directory in which the model checkpoints are saved.'
    )

    # LoRA config
    parser.add_argument('--lora_enable', default=False, action='store_true')
    parser.add_argument('--lora_r', default=32, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0., type=float)
    parser.add_argument('--lora_bias', default='none', type=str)
    parser.add_argument('--use_rslora', default=False, action='store_true')
    
    # Distributed training config
    parser.add_argument('--dist-url', default='env://', type=str) # Use env variables to set master node address and port
    parser.add_argument('--dist-backend', default='nccl', type=str) # Multi-GPU communication backend
    parser.add_argument('--horovod', default=False, action='store_true') # NOTE: Not used.
    parser.add_argument(
        '--no-set-device-rank', 
        default=False, 
        action='store_true',
        help='Disable setting device index based on local rank (when CUDA_VISIBLE_DEVICES is restricted to one per process.)'
    )
    parser.add_argument('--fsdp', default=False, action='store_true') # By default, use DDP training
    parser.add_argument(
        '--fsdp_use_orig_params',
        default=False,
        action='store_true',
        help='Passed into FSDP constructor. Enables param_groups and gradient masking for weight_decay.'
    )
    parser.add_argument('--fsdp_sharding_strategy', default='full', type=str, choices=['full', 'hybrid'])

    # W&B config
    parser.add_argument('--report_to_wandb', default=False, action='store_true')
    parser.add_argument('--save_checkpoints_to_wandb', default=False, action='store_true')

    # Quantization config
    parser.add_argument('--bits', default=16, type=int, choices=[4,8,16,32])
    parser.add_argument(
        '--precision', 
        default='amp_bf16',
        choices=['amp_bf16', 'amp_bfloat16', 'bf16', 'fp16', 'fp32'],
        help='Floating point precision.'
    )
    parser.add_argument('--double_quant', default=False, action='store_true')
    parser.add_argument('--quant_type', default='nf4', type=str)

    args, _ = parser.parse_known_args()
    train(args)