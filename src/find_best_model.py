import os
import os.path as osp
import shutil
import argparse
import pandas as pd
import yaml

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
CKPT_DIR = osp.join(ROOT_DIR, 'ckpts')
CONFIG_DIR = osp.join(ROOT_DIR, 'configs')

MED_MODELS = [
    # VLMs
    'llava-med-7b',
    'med-flamingo-9b',
    
    # LLMs
    'med42-v2-70b',
    'med42-v2-8b',
    'med42-v1-70b',
    'openbiollm-70b',
    'openbiollm-8b',
    'meditron-70b',
    'meditron-7b',
    'biomistral-7b',
    'biomedgpt-7b'
]

'''
    Miscellaneous utility functions.

'''

def remove_unneeded_ckpts(unneeded_model_paths):
    '''Removes unneeded model checkpoints.'''

    for path in unneeded_model_paths:
        if osp.exists(path):
            try:
                shutil.rmdir(path)
                logging.info(f'Removed unneeded checkpoint: "{path}".')
            except:
                logging.error(f'Failed to remove checkpoint: "{path}".')
        else:
            logging.warning(f'Checkpoint not found: "{path}".')

'''
    Functions for VLMs.

'''

def get_vlm_config_str(config, model, ft_method='lora'):
    '''Returns the config string for a given VLM.'''

    # LLaVA models
    if 'llava' in model:
        # Add LoRA configs
        if 'lora' in ft_method:
            lora_r, lora_alpha, lora_dropout, lora_bias, use_rslora = config[6:11]

            if 'qlora' in ft_method:
                config_str = 'qlora#'
            else:
                config_str = 'lora#'

            config_str += f'r={lora_r},'
            config_str += f'alpha={lora_alpha},'
            config_str += f'dropout={lora_dropout},'
            config_str += f'bias={lora_bias},'
            config_str += f'rslora={use_rslora}'
            config_str += '_'
        else:
            config_str = 'ft_'

        # Add training configs
        epochs, lr, scheduler, batch_size, decay, warmup = config[:6]
        tune_mm, freeze_mm, mm_lr, freeze_backbone = config[-4:]
        config_str += 'train#'
        config_str += f'epochs={epochs},'
        config_str += f'batch={batch_size},'
        config_str += f'lr={lr},'
        config_str += f'mm.lr={mm_lr},'
        config_str += f'decay={decay},'
        config_str += f'warmup={warmup},'
        config_str += f'scheduler={scheduler},'
        config_str += f'tune.mm={tune_mm},'
        config_str += f'freeze.mm={freeze_mm},'
        config_str += f'freeze.backbone={freeze_backbone}'

    # Flamingo models
    elif 'flamingo' in model:
        # Add training configs
        epochs, lr, scheduler, batch_size, decay, warmup, xattn, freeze_embed = config
        config_str = 'ft_'
        config_str += 'train#'
        config_str += f'epochs={epochs},'
        config_str += f'batch={batch_size},'
        config_str += f'lr={lr},'
        config_str += f'xattn={xattn},'
        config_str += f'decay={decay},'
        config_str += f'warmup={warmup},'
        config_str += f'scheduler={scheduler},'
        config_str += f'freeze.embed={freeze_embed}'

    return config_str

def find_best_vlm(model, dataset, stage_str, args):
    '''Finds the best model for a given VLM.'''

    logging.info(f'Fetching the best fine-tuned model for {model} on {dataset}...')
    model_dataset_dir = osp.join(CKPT_DIR, f'vlm/{model}/{stage_str}/{dataset}/')
    
    # LLaVA models
    if 'llava' in model:
        if args.ft_method == 'ft':
            configs = [
                (epochs, lr, scheduler, batch_size, decay, warmup,
                 tune_mm, freeze_mm, mm_lr, freeze_backbone)
                for epochs in args.epochs
                for lr in args.lr
                for scheduler in args.scheduler
                for batch_size in args.batch_size
                for decay in args.decay
                for warmup in args.warmup
                for tune_mm in args.tune_mm
                for freeze_mm in args.freeze_mm
                for mm_lr in args.mm_lr
                for freeze_backbone in args.freeze_backbone
            ]
        
        elif 'lora' in args.ft_method:
            configs = [
                (epochs, lr, scheduler, batch_size, decay, warmup, 
                 lora_r, lora_alpha, lora_dropout, lora_bias, use_rslora,
                 tune_mm, freeze_mm, mm_lr, freeze_backbone)
                for epochs in args.epochs
                for lr in args.lr
                for scheduler in args.scheduler
                for batch_size in args.batch_size
                for decay in args.decay
                for warmup in args.warmup
                for lora_r in args.lora_r
                for lora_alpha in args.lora_alpha
                for lora_dropout in args.lora_dropout
                for lora_bias in args.lora_bias
                for use_rslora in args.use_rslora
                for tune_mm in args.tune_mm
                for freeze_mm in args.freeze_mm
                for mm_lr in args.mm_lr
                for freeze_backbone in args.freeze_backbone
            ]

    # Flamingo models
    elif 'flamingo' in model:
        if args.ft_method == 'ft':
            configs = [
                (epochs, lr, scheduler, batch_size, decay, warmup, xattn, freeze_embed)
                for epochs in args.epochs
                for lr in args.lr
                for scheduler in args.scheduler
                for batch_size in args.batch_size
                for decay in args.decay
                for warmup in args.warmup
                for xattn in args.xattn
                for freeze_embed in args.freeze_embed
            ]
        
        elif 'lora' in args.ft_method:
            configs = [
                (epochs, lr, scheduler, batch_size, decay, warmup, 
                 lora_r, lora_alpha, lora_dropout, lora_bias, use_rslora,
                 xattn, freeze_embed)
                for epochs in args.epochs
                for lr in args.lr
                for scheduler in args.scheduler
                for batch_size in args.batch_size
                for decay in args.decay
                for warmup in args.warmup
                for lora_r in args.lora_r
                for lora_alpha in args.lora_alpha
                for lora_dropout in args.lora_dropout
                for lora_bias in args.lora_bias
                for use_rslora in args.use_rslora
                for xattn in args.xattn
                for freeze_embed in args.freeze_embed
            ]

    # Find best model
    best_eval_loss = None
    best_model_path = None
    unneeded_model_paths = set()

    for config in configs:
        config_str = get_vlm_config_str(config, model, ft_method=args.ft_method)
        cand_model_path = osp.join(model_dataset_dir, config_str)
        
        if 'llava' in model:
            try:
                df = pd.read_csv(osp.join(cand_model_path, 'trainer_history.csv'))
                eval_loss = df.dropna(subset=['eval_loss'])['eval_loss'].min()
            except:
                logging.error(f'History not found for {config_str}. Skipping.')
                continue
        
        elif 'flamingo' in model:
            try:
                df = pd.read_csv(osp.join(cand_model_path, 'loss_history.csv'))
                eval_loss = df['val'].min()
            except:
                logging.error(f'History not found for {config_str}. Skipping.')
                continue
            
        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_model_path = cand_model_path
        else:
            if best_eval_loss > eval_loss:
                unneeded_model_paths.add(best_model_path)
                best_eval_loss = eval_loss
                best_model_path = cand_model_path

    with open(osp.join(CONFIG_DIR, f'vlm/eval/model/{model}.yaml'), 'r') as fh:
        base_model_configs = yaml.load(fh, Loader=yaml.Loader)

    base_model_path = base_model_configs['path']
    base_model_prompt_config = base_model_configs['config']
    best_model_eval_config = dict(
        name=f'{model}_{args.ft_method}-{dataset}-best',
        base=base_model_path,
        path=best_model_path,
        config=base_model_prompt_config
    )
    best_model_eval_config_path = osp.join(
        CONFIG_DIR, f'vlm/eval/model/{model}_{args.ft_method}-{dataset}-best.yaml'
    )

    with open(best_model_eval_config_path, 'w') as fh:
        yaml.dump(best_model_eval_config, fh, default_flow_style=False, sort_keys=False)

    logging.info(f'Evaluation config saved: {best_model_eval_config_path}')

    if args.clear_ckpts:
        logging.info('Clearing up unneeded model checkpoints...')
        remove_unneeded_ckpts(unneeded_model_paths)

'''
    Functions for LLMs.

'''

def get_llm_config_str(config, ft_method='lora'):
    '''Returns the config string for a given LLM.'''

    # Add LoRA configs
    if 'lora' in ft_method:
        lora_r, lora_alpha, lora_dropout, lora_bias, use_rslora = config[-5:]

        if 'qlora' in ft_method:
            config_str = 'qlora#'
        else:
            config_str = 'lora#'

        config_str += f'r={lora_r},'
        config_str += f'alpha={lora_alpha},'
        config_str += f'dropout={lora_dropout},'
        config_str += f'bias={lora_bias},'
        config_str += f'rslora={use_rslora}'
        config_str += '_'
    else:
        config_str = 'ft_'

    # Add training configs
    epochs, lr, scheduler, batch_size, decay, warmup = config[:6]
    config_str += 'train#'
    config_str += f'epochs={epochs},'
    config_str += f'batch={batch_size},'
    config_str += f'lr={lr},'
    config_str += f'decay={decay},'
    config_str += f'warmup={warmup},'
    config_str += f'scheduler={scheduler}'

    return config_str

def find_best_llm(model, dataset, stage_str, args):
    '''Finds the best model for a given LLM.'''
    
    logging.info(f'Fetching the best fine-tuned model for {model} on {dataset}...')
    model_dataset_dir = osp.join(CKPT_DIR, f'llm/{model}/{stage_str}/{dataset}/')
    
    if args.ft_method == 'ft':
        configs = [
            (epochs, lr, scheduler, batch_size, decay, warmup)
            for epochs in args.epochs
            for lr in args.lr
            for scheduler in args.scheduler
            for batch_size in args.batch_size
            for decay in args.decay
            for warmup in args.warmup
        ]
    
    elif 'lora' in args.ft_method:
        configs = [
            (epochs, lr, scheduler, batch_size, decay, warmup, 
             lora_r, lora_alpha, lora_dropout, lora_bias, use_rslora)
            for epochs in args.epochs
            for lr in args.lr
            for scheduler in args.scheduler
            for batch_size in args.batch_size
            for decay in args.decay
            for warmup in args.warmup
            for lora_r in args.lora_r
            for lora_alpha in args.lora_alpha
            for lora_dropout in args.lora_dropout
            for lora_bias in args.lora_bias
            for use_rslora in args.use_rslora
        ]

    # Find best model
    best_eval_loss = None
    best_model_path = None
    unneeded_model_paths = set()

    for config in configs:
        config_str = get_llm_config_str(config, ft_method=args.ft_method)
        cand_model_path = osp.join(model_dataset_dir, config_str)
        try:
            df = pd.read_csv(osp.join(cand_model_path, 'trainer_history.csv'))
        except:
            logging.error(f'History not found for {config_str}. Skipping.')
            continue

        eval_loss = df.dropna(subset=['eval_loss'])['eval_loss'].min()
        
        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_model_path = cand_model_path
        else:
            if best_eval_loss > eval_loss:
                unneeded_model_paths.add(best_model_path)
                best_eval_loss = eval_loss
                best_model_path = cand_model_path

    with open(osp.join(CONFIG_DIR, f'llm/eval/model/{model}.yaml'), 'r') as fh:
        base_model_configs = yaml.load(fh, Loader=yaml.Loader)

    base_model_path = base_model_configs['path']
    attn_implementation = base_model_configs['attn_implementation']
    best_model_eval_config = dict(
        name=f'{model}_{args.ft_method}-{dataset}-best',
        base=base_model_path,
        path=best_model_path,
        attn_implementation=attn_implementation
    )
    best_model_eval_config_path = osp.join(
        CONFIG_DIR, f'llm/eval/model/{model}_{args.ft_method}-{dataset}-best.yaml'
    )

    with open(best_model_eval_config_path, 'w') as fh:
        yaml.dump(best_model_eval_config, fh, default_flow_style=False, sort_keys=False)

    logging.info(f'Evaluation config saved: {best_model_eval_config_path}')

    if args.clear_ckpts:
        logging.info('Clearing up unneeded model checkpoints...')
        remove_unneeded_ckpts(unneeded_model_paths)

def main(args):
    '''Main function for model selection.'''

    for model in args.models:
        stage_str = 'start-2,train-3' if model in MED_MODELS else 'start-0,train-3'

        for dataset in args.datasets:
            if args.model_type == 'vlm':
                find_best_vlm(model, dataset, stage_str, args)

            elif args.model_type == 'llm':
                find_best_llm(model, dataset, stage_str, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='*', default=['llava-med-7b'])
    parser.add_argument('--model_type', type=str, default='vlm', choices=['vlm','llm','bert'])
    parser.add_argument('--datasets', type=str, nargs='*', default=['vqa-rad'])
    parser.add_argument('--ft_method', help='Fine-tuning method', type=str, default='qlora', choices=['lora', 'qlora', 'ft'])
    parser.add_argument('--clear_ckpts', help='Delete unneeded checkpoints', default=False, action='store_true')
    
    # General hyperparameters
    parser.add_argument('--epochs', help='Number of training epochs', type=int, nargs='*', default=[10])
    parser.add_argument('--lr', help='Learning rate', type=float, nargs='*', default=[1e-05, 2e-05, 5e-05])
    parser.add_argument('--scheduler', help='Learning rate scheduler type', type=str, nargs='*', default=['cosine'])
    parser.add_argument('--batch_size', help='Batch size', type=int, nargs='*', default=[64])
    parser.add_argument('--decay', help='Weight decay coefficient', type=float, nargs='*', default=[0.0])
    parser.add_argument('--warmup', help='Warmup ratios', type=float, nargs='*', default=[0.03])

    # LoRA hyperparameters
    parser.add_argument('--lora_r', help='LoRA rank', type=int, nargs='*', default=[64])
    parser.add_argument('--lora_alpha', help='LoRA alpha', type=int, nargs='*', default=[16])
    parser.add_argument('--lora_dropout', help='LoRA dropout probability', type=float, nargs='*', default=[0.0])
    parser.add_argument('--lora_bias', help='LoRA bias type', type=str, nargs='*', default=['none'])
    parser.add_argument('--use_rslora', help='Rank-stabilized LoRA', type=bool, nargs='*', default=[False])

    # LLaVA-specific hyperparameters
    parser.add_argument('--tune_mm', help='Multimodal projector tuning', type=bool, nargs='*', default=[False])
    parser.add_argument('--freeze_mm', help='Multimodal projector freezing', type=bool, nargs='*', default=[True])
    parser.add_argument('--mm_lr', help='Multimodal projector learning rate', type=float, nargs='*', default=[2e-05])
    parser.add_argument('--freeze_backbone', help='LLM backbone freezing', type=bool, nargs='*', default=[False])

    # Flamingo-specific hyperparameters
    parser.add_argument('--xattn', help='Number of gated X-attention layers', type=int, nargs='*', default=[4])
    parser.add_argument('--freeze_embed', help='Chunk embedding freezing', type=bool, nargs='*', default=[False])

    args = parser.parse_args()
    
    main(args)