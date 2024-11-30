import os
import os.path as osp
import numpy as np
import json
import yaml
from datetime import datetime
import socket
import subprocess

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s | %(message)s]',
    datefmt='%d-%b-%y %H:%M:%S'
)

import hydra
from omegaconf import DictConfig, OmegaConf

# Default paths
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
TRAIN_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/vlm/train')
EVAL_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/vlm/eval')
DS_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/vlm/dist/deepspeed')

DEFAULT_SEED = 42
DEFAULT_MASTER_PORT = 29500

def format_subdir(model_args, train_args, dataset_args, gpu_ids):
    '''
        Dynamically formats the Hydra log subdirectory name.
    
        Example: 'qlora#r=128,alpha=256,dropout=0,bias=none_train#epochs=10,batch=16,...'

    '''
    
    # LLaVA models
    if 'llava' in model_args.name:
        if train_args.train_stage == 3:
            subdir = f'{dataset_args.name}/'
        else:
            subdir = ''
        
        # Add LoRA configs
        if train_args.lora_enable:
            if train_args.bits in [4,8]:
                subdir += 'qlora#'
            else:
                subdir += 'lora#'
        
            subdir += f'r={train_args.lora_r},'
            subdir += f'alpha={train_args.lora_alpha},'
            subdir += f'dropout={train_args.lora_dropout},'
            subdir += f'bias={train_args.lora_bias},'
            subdir += f'rslora={train_args.use_rslora}'
            subdir += '_'
        else:
            subdir += 'ft_'
        
        # Add training configs
        subdir += 'train#'
        subdir += f'epochs={train_args.num_train_epochs},'
        subdir += f'batch={train_args.per_device_train_batch_size * len(gpu_ids)},'
        subdir += f'lr={train_args.learning_rate},'
        subdir += f'mm.lr={train_args.mm_projector_lr},'
        subdir += f'decay={train_args.weight_decay},'
        subdir += f'warmup={train_args.warmup_ratio},'
        subdir += f'scheduler={train_args.lr_scheduler_type},'
        subdir += f'tune.mm={train_args.tune_mm_mlp_adapter},'
        subdir += f'freeze.mm={train_args.freeze_mm_mlp_adapter},'
        subdir += f'freeze.backbone={train_args.freeze_backbone}'

    # Open-Flamingo models
    elif 'flamingo' in model_args.name:
        if train_args.train_stage == 3:
            subdir = f'{dataset_args.name}/'
        else:
            subdir = ''

        # Add LoRA configs
        if train_args.lora_enable:
            if train_args.bits in [4,8]:
                subdir += 'qlora#'
            else:
                subdir += 'lora#'
        
            subdir += f'r={train_args.lora_r},'
            subdir += f'alpha={train_args.lora_alpha},'
            subdir += f'dropout={train_args.lora_dropout},'
            subdir += f'bias={train_args.lora_bias},'
            subdir += f'rslora={train_args.use_rslora}'
            subdir += '_'
        else:
            subdir += 'ft_'

        # Add training configs
        subdir += 'train#'
        subdir += f'epochs={train_args.num_epochs},'
        subdir += f'batch={train_args.batch_size},'
        subdir += f'lr={train_args.learning_rate},'
        subdir += f'xattn={model_args.cross_attn_every_n_layers},'
        subdir += f'decay={train_args.weight_decay},'
        subdir += f'warmup={train_args.warmup_ratio},'
        subdir += f'scheduler={train_args.lr_scheduler},'
        subdir += f'freeze.embed={train_args.freeze_lm_embeddings}'
    
    return subdir

def update_model_path(model_root_dir, model_path):
    '''Updates the model path if it is local.'''

    if osp.exists(osp.join(model_root_dir, model_path)):
        return osp.join(model_root_dir, model_path)
    else:
        return model_path

OmegaConf.register_new_resolver('format_subdir', format_subdir)
OmegaConf.register_new_resolver('update_model_path', update_model_path)

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

def port_in_use(port):
    '''Checks if a port is in use.'''

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def run_subprocess(command, env=None):
    '''Executes a subprocess, constantly printing the output.'''

    popen = subprocess.Popen(
        command, 
        bufsize=1,
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True,
        env=env
    )

    for line in popen.stdout:
        print(line, end='')

@hydra.main(version_base=None, config_path=TRAIN_CONFIG_DIR, config_name='train-config')
def main(args: DictConfig) -> None:
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    start_time = datetime.now()

    if 'llava' in args.model.name:
        subprocess_script = f'{ROOT_DIR}/src/vlm/llava_models/train_llava.py'

        # Collect all Hydra configs into a list of arguments for the subprocess
        hydra_args = []
        deepspeed_id = None
        gpu_ids = None
        for arg, value in args.items():
            if arg == 'gpu_ids':
                gpu_ids = [str(val) for val in value]
                continue

            elif arg == 'deepspeed_id':
                deepspeed_id = value
                continue

            if isinstance(value, (dict, DictConfig)):
                for k,v in value.items():
                    if k == 'gpu_ids':
                        gpu_ids = [str(val) for val in v]
                        continue

                    elif k == 'deepspeed_id':
                        deepspeed_id = v
                        continue

                    if (arg in ['model', 'dataset']) and k == 'name':
                        k = f'{arg}_{k}'

                    hydra_args.append(f'--{k}')
                    hydra_args.append(str(v)) # Need to convert to string for subprocess
            else:
                hydra_args.append(f'--{arg}')
                hydra_args.append(str(value))

        # Set up DeepSpeed arguments
        command = ['deepspeed']
        command += ['-i', f'localhost:{",".join(gpu_ids)}']
        
        # Add DeepSpeed masterport
        master_port = DEFAULT_MASTER_PORT
        found_open_port = False
        while not found_open_port:
            if port_in_use(master_port):
                master_port += 1
            else:
                found_open_port = True

        command += ['--master_port', str(master_port)]

        # Add subprocess arguments    
        command += [subprocess_script]
        command += hydra_args

        # Add DeepSpeed config
        command += ['--deepspeed', osp.join(DS_CONFIG_DIR, f'{deepspeed_id}.json')]
        
        # Run training as subprocess
        run_subprocess(command)

    elif 'flamingo' in args.model.name:
        subprocess_script = f'{ROOT_DIR}/src/vlm/flamingo_models/train_flamingo.py'

        # Collect all Hydra configs into a list of arguments for the subprocess
        hydra_args = []
        gpu_ids = None
        for arg, value in args.items():
            if arg == 'gpu_ids':
                gpu_ids = [str(val) for val in value]
                continue

            if isinstance(value, (dict, DictConfig)):
                for k,v in value.items():
                    if k == 'gpu_ids':
                        gpu_ids = [str(val) for val in v]
                        continue

                    if (arg in ['model', 'dataset']) and k == 'name':
                        k = f'{arg}_{k}'

                    if type(v) == bool:
                        if v:
                            hydra_args.append(f'--{k}')
                    else:
                        hydra_args.append(f'--{k}')
                        hydra_args.append(str(v)) # Need to convert to string for subprocess
            else:
                if type(value) == bool:
                    if value:
                        hydra_args.append(f'--{arg}')
                else:
                    hydra_args.append(f'--{arg}')
                    hydra_args.append(str(value))

        # Add GPU specification arguments
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
        
        # Set up torchrun arguments
        command = ['torchrun']
        command += ['--nnodes', '1']
        command += ['--nproc_per_node', str(len(gpu_ids))]
        
        # Add torchrun masterport
        master_port = DEFAULT_MASTER_PORT
        found_open_port = False
        while not found_open_port:
            if port_in_use(master_port):
                master_port += 1
            else:
                found_open_port = True

        command += ['--master_port', str(master_port)]

        # Add subprocess arguments    
        command += [subprocess_script]
        command += hydra_args

        # Run training as subprocess
        run_subprocess(command, env=env)

    else:
        raise NotImplementedError(f'Training for "{args.model.name}" not yet supported.')

    logging.info(f'Elapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    main()