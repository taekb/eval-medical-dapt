import os.path as osp
import socket
import subprocess

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

import hydra
from omegaconf import DictConfig, OmegaConf

# Default paths
ROOT_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
EVAL_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/vlm/eval')
DIST_CONFIG_DIR = osp.join(ROOT_DIR, 'configs/vlm/dist/accelerate')

DEFAULT_MASTER_PORT = 29500

def format_subdir(
    temperature,
    prompt_type,
    n_shot,
    constrain_vocab,
    predict_with_logprob,
    n_seeds
):
    '''
        Dynamically formats the Hydra log subdirectory name.

    '''

    subdir = f'T={temperature},'
    subdir += f'prompt={f"{n_shot}-shot" if prompt_type == "few-shot" else "zero-shot"},'
    subdir += f'constrain_vocab={constrain_vocab},'
    subdir += f'pred_logprob={predict_with_logprob},'
    subdir += f'n_seeds={n_seeds}'

    return subdir

OmegaConf.register_new_resolver('format_subdir', format_subdir)

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

@hydra.main(version_base=None, config_path=EVAL_CONFIG_DIR, config_name='eval-config')
def main(args: DictConfig) -> None:
    logging.info(f'Running inference with "{args.model.name}" on "{args.dataset.name}"...')
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_path = osp.join(log_dir, f'{args.eval_split}_results.pkl')

    # Check if evaluation results already exist
    # NOTE: Overridden if force_eval is True
    if osp.exists(results_path) and not args.force_eval:
        logging.info(
            f'Evaluation results already saved: {osp.join(log_dir, f"{args.eval_split}_results.pkl")}'
        )
    
    # If not, load checkpoint and evaluate model
    else:       
        # Set up GPU indices
        gpu_ids = args.gpu_ids
        
        # Temporarily save DictConfig
        config_path = osp.join(log_dir, 'temp_config.yaml')
        with open(config_path, 'w') as fh:
            fh.write(OmegaConf.to_yaml(args))

        if 'llava' in args.model.path.lower():
            subprocess_script = f'{ROOT_DIR}/src/vlm/llava_models/infer_llava.py'
        
        elif 'flamingo' in args.model.path.lower():
            subprocess_script = f'{ROOT_DIR}/src/vlm/flamingo_models/infer_flamingo.py'

        # Set up command-line arguments to pass
        accelerate_config_path = osp.join(DIST_CONFIG_DIR, 'inference.yaml')

        command = ['accelerate', 'launch']
        command += ['--config_file', accelerate_config_path]
        command += ['--gpu_ids', ','.join([str(i) for i in gpu_ids])]
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
        command += ['--log_dir', log_dir]
        command += ['--config_path', config_path]

        # Run inference as subprocess
        run_subprocess(command)

if __name__ == '__main__':
    main()