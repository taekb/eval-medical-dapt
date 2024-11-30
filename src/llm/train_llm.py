import os
import os.path as osp
import numpy as np
from datetime import datetime
import socket
import subprocess
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

import hydra
from omegaconf import DictConfig, OmegaConf

# Default paths
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
TRAIN_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/llm/train')
DIST_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/llm/dist')

DEFAULT_SEED = 42
DEFAULT_MASTER_PORT = 29500

def format_subdir(train_args, dataset_args, n_nodes, gpu_ids):
    '''
        Dynamically formats the Hydra log subdirectory name.
    
        Example: 'qlora#r=128,alpha=256,dropout=0,bias=none_train#epochs=10,batch=16,...'

    '''
    
    if train_args.train_stage == 3:
        subdir = f'{dataset_args.name}/'
    else:
        subdir = ''

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
        subdir = 'ft_'

    # Add training configs
    subdir += 'train#'
    subdir += f'epochs={train_args.num_train_epochs},'
    subdir += f'batch={n_nodes * train_args.per_device_train_batch_size * len(gpu_ids)},'
    subdir += f'lr={train_args.learning_rate},'
    subdir += f'decay={train_args.weight_decay},'
    subdir += f'warmup={train_args.warmup_ratio},'
    subdir += f'scheduler={train_args.lr_scheduler_type}'
    
    return subdir

def update_model_path(model_root_dir, model_path):
    '''Updates the model path if it is local.'''

    if osp.exists(osp.join(model_root_dir, model_path)):
        return osp.join(model_root_dir, model_path)
    else:
        return model_path
    
def update_data_path(data_root_dir, data_path):
    '''Updates the dataset path if it is local.'''

    if osp.exists(osp.join(data_root_dir, data_path)):
        return osp.join(data_root_dir, data_path)
    else:
        return data_path

OmegaConf.register_new_resolver('format_subdir', format_subdir)
OmegaConf.register_new_resolver('update_model_path', update_model_path)
OmegaConf.register_new_resolver('update_data_path', update_data_path)

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

    subprocess_script = f'{ROOT_DIR}/src/llm/_train_llm.py'

    # Collect all Hydra configs into a list of arguments for the subprocess
    hydra_args = []
    gpu_ids = None

    for arg, value in args.items():
        if arg == 'gpu_ids':
            gpu_ids = [str(val) for val in value]
            continue

        elif arg == 'n_nodes':
            n_nodes = str(value)
            continue

        elif arg == 'head_node_ip':
            head_node_ip = value
            continue
        
        if isinstance(value, (dict, DictConfig)):
            for k,v in value.items():
                if k == 'gpu_ids':
                    gpu_ids = [str(val) for val in v]
                    continue

                elif k == 'n_nodes':
                    n_nodes = str(v)
                    continue

                elif k == 'head_node_ip':
                    head_node_ip = v
                    continue

                if (arg in ['model', 'dataset']) and k == 'name':
                    k = f'{arg}_{k}'

                if type(v) == bool:
                    if v:
                        hydra_args.append(f'--{k}')
                else:
                    hydra_args.append(f'--{k}')
                    hydra_args.append(str(v))
        else:
            if type(value) == bool:
                if value:
                    hydra_args.append(f'--{arg}')
            else:
                hydra_args.append(f'--{arg}')
                hydra_args.append(str(value))

    # Set up accelerate arguments
    if args.zero_stage == 2:
        accelerate_config_path = osp.join(DIST_CONFIG_DIR, 'training-zero2.yaml')
    elif args.zero_stage == 3:
        accelerate_config_path = osp.join(DIST_CONFIG_DIR, 'training-zero3.yaml')
    else:
        logging.error(f'ZeRO config not set up for specified stage: "{args.zero_stage}"')
        raise ValueError

    command = ['accelerate', 'launch']
    command += ['--config_file', accelerate_config_path]

    if int(n_nodes) > 1:
        if head_node_ip is None:
            logging.error('Head node IP not specified for multi-node training.')
            raise ValueError

        command += ['--num_machines', n_nodes]
        command += ['--num_processes', str(int(n_nodes) * len(gpu_ids))]
        command += ['--main_process_ip', head_node_ip]
    else:
        command += ['--gpu_ids', ','.join(gpu_ids)]
        command += ['--num_processes', str(len(gpu_ids))]

    # Add masterport
    master_port = DEFAULT_MASTER_PORT
    found_open_port = False
    while not found_open_port:
        if port_in_use(master_port):
            master_port += 1
        else:
            found_open_port = True

    command += ['--main_process_port', str(master_port)]

    # Add subprocess arguments
    command += [subprocess_script]
    command += hydra_args

    # Run training as subprocess
    run_subprocess(command)

    logging.info(f'Elapsed: {str(datetime.now() - start_time)}')

if __name__ == '__main__':
    main()