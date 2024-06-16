import os
import os.path as osp
import json
import abc
from collections.abc import Iterable
import numpy as np
import re
from functools import partial
from typing import Tuple, Optional, Union, Dict, List
from jinja2 import Template
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s | %(message)s]',
    datefmt='%d-%b-%y %H:%M:%S'
)

from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

from sklearn.model_selection import train_test_split

from datasets import load_dataset

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizer

# Default paths
ROOT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
DATA_DIR = '/data'

IGNORE_INDEX = -100

DEFAULT_SEED = 42

N_MCQ = {
    'medqa': 5,
    'medmcqa': 4,
    'pubmedqa': 3,
    'mmlu_anatomy': 4,
    'mmlu_clinical-knowledge': 4,
    'mmlu_college-biology': 4,
    'mmlu_college-medicine': 4,
    'mmlu_medical-genetics': 4,
    'mmlu_professional-medicine': 4,
    'mmlu_high-school-biology': 4,
    'mmlu_virology': 4,
    'mmlu_nutrition': 4
}

# Models that use a chat template
CHAT_MODELS = [
    'aaditya/Llama3-OpenBioLLM-70B',
    'aaditya/Llama3-OpenBioLLM-8B',
    'meta-llama/Meta-Llama-3-70B-Instruct',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    'meta-llama/Llama-2-7b-chat-hf',
    'mistralai/Mistral-7B-Instruct-v0.1',

    'openbiollm-70b',
    'openbiollm-8b',
    'llama-3-70b',
    'llama-2-7b-chat',
    'mistral-7b-v0.1'
]

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

'''
    QA dataset classes and utility functions.

'''

def qa_collate_fn(qa_samples, pad_token_id, return_dict=False):
    '''Collate function for batching QA samples.'''

    input_id_list, label_id_list = zip(*qa_samples)

    # Stack the input_ids and label_ids together
    batch_input_ids = pad_sequence(
        [ids[0] for ids in input_id_list],
        batch_first=True,
        padding_value=pad_token_id
    )
    batch_label_ids = pad_sequence(
        [ids[0] for ids in label_id_list],
        batch_first=True,
        padding_value=pad_token_id
    )
    batch_label_ids[batch_label_ids == pad_token_id] = IGNORE_INDEX

    # Get the attention masks
    batch_attention_masks = batch_input_ids.ne(pad_token_id)

    if return_dict:
        batch = dict(
            input_ids=batch_input_ids,
            labels=batch_label_ids,
            attention_mask=batch_attention_masks
        )
        return batch
    else:
        return (batch_input_ids, batch_label_ids, batch_attention_masks)

class QADataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    '''Base class for QA datasets.'''

    def __init__(
        self, 
        name: str, 
        qa_dir: str, 
        splits: List[str],
        main_split: Optional[str] = None,
        seed: Optional[int] = 42, # Seed for sampling few-shot examples
        verbose: Optional[bool] = False,
        hf_cache_dir: Optional[str] = None,
        **kwargs
    ) -> None:

        self.name = name
        self.qa_dir = qa_dir
        self.splits = splits
        self.qa_dict = {}
        self.qas = {}
        self.seed = seed
        self.verbose = verbose
        self.hf_cache_dir = hf_cache_dir
        self.tokenized = False

        if main_split is not None:
            self.main_split = main_split
        elif len(self.splits) == 1:
            self.main_split = self.splits[0]
        else:
            logging.error('Main split to use must be specified if len(split) > 1.')
            raise ValueError

    @abc.abstractmethod
    def load_qas(self) -> None:
        raise NotImplementedError
    
    def load_prompt_template(
        self, 
        model_name: str, 
        dataset_name: str, 
        prompt_type: str, 
        sample_kwargs: Optional[Dict] = None
    ) -> None:
        '''Load the prompt templates to use for prompting.'''
        
        from llm.utils import (
            sample_llm_system_prompt,
            get_llm_system_prompt, 
            sample_llm_prompt_template,
            get_llm_prompt_template
        )

        # Load default system prompt and template
        if sample_kwargs is None:
            self.system_prompt = get_llm_system_prompt(model_name, dataset_name)
            self.prompt_template = get_llm_prompt_template(model_name, prompt_type=prompt_type)

        # Otherwise, randomly sample; unless the seed is None
        else:
            n_mcq = N_MCQ[dataset_name]

            if sample_kwargs['system_prompt_seed'] is None:
                self.system_prompt = get_llm_system_prompt(model_name, dataset_name)
            else:
                self.system_prompt = sample_llm_system_prompt(seed=sample_kwargs['system_prompt_seed']).format(chr(ord('A')+n_mcq-1))

            if sample_kwargs['prompt_template_seed'] is None:
                self.prompt_template = get_llm_prompt_template(model_name, prompt_type=prompt_type)
            else:
                self.prompt_template = sample_llm_prompt_template(seed=sample_kwargs['prompt_template_seed'])
    
    def load_and_apply_prompt_template(
        self,
        model_name: str,
        sample_kwargs: Optional[Dict] = None,
        prompt_type: Optional[str] = 'zero-shot',
        system_prompt: Optional[str] = None,
        prompt_template: Optional[Union[Template,Tuple[Template]]] = None,
        tokenize: Optional[bool] = False,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        '''Loads and applies the selected/sampled prompt template to the data.'''

        from llm.utils import allows_system_prompts

        if tokenize:
            if tokenizer is None:
                logging.error('Tokenizer must be provided for tokenization mode.')
                raise ValueError
                
            self.qas[self.main_split] = {'input_ids': [], 'label_ids': []}
        else:
            self.qas[self.main_split] = []

        # Parse model name to handle fine-tuned models
        if any([ft in model_name for ft in ['lora', 'ft']]):
            parsed_model_name = model_name.split('_')[0]
        else:
            parsed_model_name = model_name

        # Load the prompt template
        if system_prompt is not None and prompt_template is not None:
            self.system_prompt = system_prompt
            self.prompt_template = prompt_template
        else:
            self.load_prompt_template(
                parsed_model_name, 
                self.name,
                prompt_type, 
                sample_kwargs=sample_kwargs
            )

        few_shot_qas = getattr(self, 'few_shot_qas', None)
        qas = self.qa_dict[self.main_split]
        pbar = tqdm(qas, desc=f'Applying template to "{self.main_split}" split', unit='QA pair', disable=(not self.verbose), total=len(qas))
        for (question, options, answer) in pbar:
            # Models with chat templates provided via the tokenizer
            if parsed_model_name in CHAT_MODELS:
                # Load question and answer templates
                q_template = self.prompt_template['question']
                a_template = self.prompt_template['answer']
                conv = []

                if allows_system_prompts(tokenizer) and prompt_type != 'zero-shot-ft':
                    conv.append({'role': 'system', 'content': self.system_prompt})
                
                # Add few-shot examples to conversation
                if few_shot_qas is not None:
                    for i, (fs_question, fs_options, fs_answer) in enumerate(few_shot_qas):
                        fs_q_prompt = q_template.render(question=fs_question, options=fs_options)
                        fs_a_prompt = a_template.render(options=fs_options, answer=fs_answer)
                        
                        if i == 0 and not allows_system_prompts(tokenizer) and prompt_type != 'zero-shot-ft':
                            fs_q_prompt = f'{self.system_prompt}\n{fs_question}'

                        conv.append({'role': 'user', 'content': fs_q_prompt})
                        conv.append({'role': 'assistant', 'content': fs_a_prompt})
                    
                # Add QA pair
                q_prompt = q_template.render(question=question, options=options)
                conv.append({'role': 'user', 'content': q_prompt})

                if tokenize:
                    # Add generation header for Llama-3 and OpenBioLLM models
                    if any([m in parsed_model_name for m in ['llama-3', 'openbiollm']]):
                        context_len = len(tokenizer.apply_chat_template(conv, return_tensors='pt', add_generation_prompt=True)[0])
                    elif 'mistral-7b-v0.1' in parsed_model_name:
                        context_len = len(tokenizer.apply_chat_template(conv, return_tensors='pt')[0]) - 1
                    else:
                        context_len = len(tokenizer.apply_chat_template(conv, return_tensors='pt')[0])

                    a_prompt = a_template.render(options=options, answer=answer)
                    conv.append({'role': 'assistant', 'content': a_prompt})
                    input_ids = tokenizer.apply_chat_template(
                        conv,
                        return_tensors='pt',
                        truncation=True,
                        max_length=tokenizer.model_max_length
                    )
                    label_ids = input_ids.clone()
                    label_ids[0,:context_len] = IGNORE_INDEX

                    self.qas[self.main_split]['input_ids'].append(input_ids)
                    self.qas[self.main_split]['label_ids'].append(label_ids)

                else:
                    a_prompt = a_template.render(options=options, answer=None)
                    conv.append({'role': 'assistant', 'content': a_prompt})
                    prompt = tokenizer.apply_chat_template(conv, tokenize=False)

                    if prompt.strip().endswith(tokenizer.eos_token):
                        prompt = prompt[:prompt.rindex(tokenizer.eos_token)]

                    if prompt.strip().endswith('<|eot_id|>'):
                        prompt = prompt[:prompt.rindex('<|eot_id|>')]

                    self.qas[self.main_split].append(prompt)
            
            # Models without any chat template
            else:
                context = self.prompt_template['full'].render(
                    system_prompt=(None if prompt_type == 'zero-shot-ft' else self.system_prompt),
                    few_shot_qas=few_shot_qas,
                    qa=(question, options, None)
                )

                if tokenize:
                    prompt = self.prompt_template['full'].render(
                        system_prompt=(None if prompt_type == 'zero-shot-ft' else self.system_prompt),
                        few_shot_qas=few_shot_qas,
                        qa=(question, options, answer)
                    )
                    context = tokenizer(context, return_tensors='pt').input_ids[0]
                    context_len = len(context)

                    if parsed_model_name in ['llama-3-8b', 'llama-2-70b', 'llama-2-7b', 'meditron-70b', 'meditron-7b']:
                        context_len -= 1
                    
                    prompt += tokenizer.eos_token
                    
                    tokenized_text = tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=tokenizer.model_max_length
                    )
                    input_ids = tokenized_text.input_ids
                    label_ids = input_ids.clone()
                    label_ids[0,:context_len] = IGNORE_INDEX
                    
                    self.qas[self.main_split]['input_ids'].append(input_ids)
                    self.qas[self.main_split]['label_ids'].append(label_ids)
                
                else:
                    self.qas[self.main_split].append(context)

        if tokenize:
            self.tokenized = True

    def sample_few_shot_qas(self, n_shot: Optional[int] = 5, seed: Optional[int] = None) -> None:
        '''Randomly sample k QA pairs from the training set for few-shot prompting.'''

        if 'train' not in self.qa_dict:
            logging.error('Training set must be loaded before sampling few-shot examples.')
            raise RuntimeError
        
        # Randomly sample k QA pairs
        # NOTE: We follow the behavior expected for HF dataset shuffle()
        # Reference: https://github.com/huggingface/datasets/blob/2.19.0/src/datasets/arrow_dataset.py#L4272
        n_train = len(self.qa_dict['train'])
        few_shot_idxs = np.arange(n_train)
        generator = np.random.default_rng(seed if seed is not None else self.seed)
        permutation = generator.permutation(n_train)
        few_shot_idxs = few_shot_idxs[permutation][:n_shot]
        self.few_shot_qas = [self.qa_dict['train'][i] for i in few_shot_idxs]

    def __len__(self) -> int:
        return len(self.qa_dict[self.main_split])

    def __getitem__(self, idx: int) -> Union[Tuple[str,str,str,str], Tuple[torch.Tensor,torch.Tensor]]:
        if self.tokenized:
            input_id_list = self.qas[self.main_split]['input_ids']
            label_id_list = self.qas[self.main_split]['label_ids']

            if type(idx) == int:
                return (input_id_list[idx], label_id_list[idx])
            
            elif isinstance(idx, Iterable):
                return [(input_id_list[i], label_id_list[i]) for i in idx]
        
            elif isinstance(idx, slice):
                start = idx.start
                stop = idx.stop
                step = idx.step if idx.step is not None else 1
                idx = range(start, stop, step)

                return [(input_id_list[i], label_id_list[i]) for i in idx]
        else:
            return self.qas[self.main_split][idx]
        
class MedQADataset(QADataset):
    '''Dataset class for MedQA (Jin et al., 2021).'''

    def __init__(self, name='medqa', qa_dir='bigbio/med_qa', **kwargs):
        super(MedQADataset, self).__init__(name=name, qa_dir=qa_dir, **kwargs)
        self.load_qas()

    def load_qas(self) -> None:
        hf_dataset = load_dataset(self.qa_dir, cache_dir=self.hf_cache_dir, trust_remote_code=True)
        
        for split in self.splits:
            dataset = hf_dataset['validation' if split == 'val' else split]

            questions = []
            answers = []
            options = []

            pbar = tqdm(dataset, desc=f'Loading the "{split}" split', unit='QA pair', disable=(not self.verbose), total=len(dataset))
            for qa_sample in pbar:
                questions.append(qa_sample['question'].strip())
                options.append([o['value'] for o in qa_sample['options']])

                # Check if the answer is indeed in the options
                if qa_sample['answer'] in options[-1]:
                    answers.append(qa_sample['answer'])
                else:
                    answer_key = qa_sample['answer_idx']
                    for o in qa_sample['options']:
                        if o['key'] == answer_key:
                            answers.append(o['value'])
                            break

            self.qa_dict[split] = list(zip(questions, options, answers))

class MedMCQADataset(QADataset):
    '''Dataset class for MedMCQA (Pal et al., 2022).'''

    def __init__(self, name='medmcqa', qa_dir='openlifescienceai/medmcqa', **kwargs):
        super(MedMCQADataset, self).__init__(name=name, qa_dir=qa_dir, **kwargs)
        self.load_qas()

    # Handles MedMCQA answer choices that are poorly preprocessed
    def clean_option(self, option) -> str:
        # 2009
        if option == 'c)\tCannot be separated by probe':
            option = 'Cannot be separated by probe'
        
        elif option == 'd)\tIs marginal gingiva':
            option = 'Is marginal gingiva'

        # 3166
        elif option == 'A. ii) B. ii) C. i) D. i) E. i)':
            option = 'A: (ii), B: (ii), C: (i), D: (i), E: (i)'
        
        elif option == 'A. i) B. ii) C. ii) D. i) E. i)':
            option = 'A: (i), B: (ii), C: (ii), D: (i), E: (i)'

        elif option == 'A. i) B. ii) C. i) D. i) E. ii)':
            option = 'A: (i), B: (ii), C: (i), D: (i), E: (ii)'

        elif option == 'A. i) B. ii) C. i) D. i) E. i)':
            option = 'A: (i), B: (ii), C: (i), D: (i), E: (i)'

        # 3755
        elif option == 'a)\tTo balance the denture':
            option = 'To balance the denture'

        elif option == 'b)\tTo act as a direct retainer':
            option = 'To act as a direct retainer'

        elif option == 'c)\tTo counteract the movement of denture which is caused during engagement of retentive arm':
            option = 'To counteract the movement of denture which is caused during engagement of retentive arm'
        
        elif option == 'd)\tNone of the above':
            option = 'None of the above'

        return option
    
    # NOTE: MedMCQA does not have a public test set (answers are hidden).
    # We therefore use the validation set for testing, as in PMC-LLaMA (Wu et al., 2023) and BioMistral (Labrak et al., 2024).
    # We take a random 80-20 split on the training set for validation. 
    def load_qas(self) -> None:
        hf_dataset = load_dataset(self.qa_dir, cache_dir=self.hf_cache_dir, trust_remote_code=True)
        
        for split in self.splits:
            dataset = hf_dataset['validation' if split == 'test' else 'train']

            if split != 'test':
                dataset = dataset.train_test_split(test_size=0.2, seed=DEFAULT_SEED)
                dataset = dataset['train'] if split == 'train' else dataset['test']
            
            questions = []
            answers = []
            options = []

            pbar = tqdm(dataset, desc=f'Loading the "{split}" split', unit='QA pair', disable=(not self.verbose), total=len(dataset))
            for qa_sample in pbar:
                questions.append(qa_sample['question'].strip())
                options.append([
                    self.clean_option(qa_sample['opa']), 
                    self.clean_option(qa_sample['opb']), 
                    self.clean_option(qa_sample['opc']), 
                    self.clean_option(qa_sample['opd'])
                ])
                answers.append(options[-1][qa_sample['cop']])
                
            self.qa_dict[split] = list(zip(questions, options, answers))         

class MMLUMedicalDataset(QADataset):
    '''Dataset class for medical datasets in MMLU (Hendrycks et al., 2021).'''

    def __init__(self, name='mmlu_college-medicine', qa_dir='lukaemon/mmlu', **kwargs):
        medical_dataset_names = [
            'anatomy', 
            'clinical-knowledge', 
            'college-biology', 
            'college-medicine',
            'high-school-biology', 
            'medical-genetics', 
            'nutrition',
            'professional-medicine',
            'virology'
        ]

        if name.split('_')[-1] not in medical_dataset_names:
            logging.warning(f'Dataset "{name}" is not one of the 9 medical datasets in MMLU.')
        
        super(MMLUMedicalDataset, self).__init__(name=name, qa_dir=qa_dir, **kwargs)
        self.load_qas()

    def load_qas(self) -> None:
        hf_dataset = load_dataset(
            self.qa_dir,
            self.name.split('_')[-1].replace('-', '_'),
            cache_dir=self.hf_cache_dir,
            trust_remote_code=True
        )

        for split in self.splits:
            dataset = hf_dataset['validation' if split == 'val' else split]

            questions = []
            answers = []
            options = []

            pbar = tqdm(dataset, desc=f'Loading the "{split}" split', unit='QA pair', disable=(not self.verbose), total=len(dataset))
            for qa_sample in pbar:
                questions.append(qa_sample['input'].strip())
                answers.append(qa_sample[qa_sample['target']])
                options.append([qa_sample['A'], qa_sample['B'], qa_sample['C'], qa_sample['D']])

            self.qa_dict[split] = list(zip(questions, options, answers))  

class PubMedQADataset(QADataset):
    '''Dataset class for PubMedQA (Jin et al., 2019).'''

    def __init__(self, name='pubmedqa', qa_dir='bigbio/pubmed_qa', **kwargs):
        super(PubMedQADataset, self).__init__(name=name, qa_dir=qa_dir, **kwargs)
        self.load_qas()

    def load_qas(self) -> None:
        # NOTE: We use the splits by prior works (e.g., Labrak et al., 2024; Singhal et al., 2023).
        # NOTE: Splits are provided by BigBio (Fries et al., 2022).

        for split in self.splits:
            # Use 211k artificially generated QA samples for training
            if split == 'train':
                hf_dataset = load_dataset(
                    self.qa_dir,
                    'pubmed_qa_artificial_bigbio_qa',
                    split='train+validation',
                    cache_dir=self.hf_cache_dir,
                    trust_remote_code=True
                )
                dataset = hf_dataset
            
            # Use 500 out of 1000 expert-labeled QA samples for validation
            elif split == 'val':
                hf_dataset = load_dataset(
                    self.qa_dir,
                    'pubmed_qa_labeled_fold0_bigbio_qa',
                    split='train+validation',
                    cache_dir=self.hf_cache_dir,
                    trust_remote_code=True
                )
                dataset = hf_dataset

            # Use remaining 500 out of 1000 expert-labeled QA samples for testing
            elif split == 'test':
                hf_dataset = load_dataset(
                    self.qa_dir, 
                    'pubmed_qa_labeled_fold0_bigbio_qa',
                    cache_dir=self.hf_cache_dir,
                    trust_remote_code=True
                )
                dataset = hf_dataset['test']

            questions = []
            answers = []
            options = []

            pbar = tqdm(dataset, desc='Preprocessing QA pairs', unit='QA pair', disable=(not self.verbose), total=len(dataset))
            for qa_sample in pbar:
                context = qa_sample['context'].strip()
                question = qa_sample['question'].strip()
                question = f'{context}\n\n{question}'

                questions.append(question)
                answers.append(qa_sample['answer'][0])
                options.append(qa_sample['choices'])

            self.qa_dict[split] = list(zip(questions, options, answers))

'''
    VQA dataset classes and utility functions.

'''

def vqa_collate_fn(qa_samples, tokenizer):
    '''Collate function for batching VQA samples.'''

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
        padding_value=tokenizer.pad_token_id
    )
    batch_label_ids[batch_label_ids == tokenizer.pad_token_id] = IGNORE_INDEX

    # Get the attention masks
    batch_attention_masks = batch_input_ids.ne(tokenizer.pad_token_id)

    return (batch_image_tensor, batch_input_ids, batch_label_ids, batch_attention_masks)

class VQADataset(torch.utils.data.Dataset):
    '''Base class for VQA datasets.'''

    def __init__(
        self,
        name: str, 
        qa_dir: str, 
        image_dir: str, 
        splits: List[str],
        main_split: Optional[str] = None,
        seed: Optional[int] = 42, # Seed for sampling few-shot examples
        verbose: Optional[bool] = False,
        **kwargs
    ) -> None:
        
        self.name = name
        self.qa_dir = qa_dir
        self.image_dir = image_dir
        self.splits = splits
        self.qa_dict = {}
        self.qas = {}
        self.images = {}
        self.seed = seed
        self.verbose = verbose

        if main_split is not None:
            self.main_split = main_split
        elif len(self.splits) == 1:
            self.main_split = self.splits[0]
        else:
            logging.error('Main split to use must be specified if len(split) > 1.')
            raise ValueError
        
        self.load_qas()
        
    def load_qas(self) -> None:
        '''Loads and preprocesses the given question-answer pairs.'''

        for split in self.splits:
            try:
                qa_path = osp.join(self.qa_dir, f'{split}.jsonl')
                qa_data = read_jsonl(qa_path)
            except:
                if split == 'val':
                    qa_path = osp.join(self.qa_dir, 'train.jsonl')
                    qa_data = read_jsonl(qa_path)
                    _, val_idx = train_test_split(np.arange(len(qa_data)), test_size=0.2, random_state=self.seed)
                    qa_data = [qa_data[i] for i in val_idx]
                else:
                    logging.error(f'QA data for "{split}" split not found.')
                    raise FileNotFoundError

            questions = []
            answers = []
            image_paths = []
            options = []

            pbar = tqdm(qa_data, desc=f'Loading the "{split}" split', unit='QA pair', disable=(not self.verbose), total=len(qa_data))
            for qa in pbar:
                questions.append(qa['conversations'][0]['value'])
                options.append(qa['options'])
                answers.append(qa['conversations'][1]['value'])

                try:
                    image_paths.append([osp.join(self.image_dir, qa['image'])])
                except:
                    image_paths.append([osp.join(self.image_dir, f'{split}/{qa["image"]}')])

            self.qa_dict[split] = list(zip(questions, options, answers, image_paths))
    
    def load_prompt_template(
        self,
        model_name: str,
        dataset_name: str,
        prompt_type: str,
        sample_kwargs: Optional[Dict] = None
    ) -> None:
        '''Load the prompt templates to use for prompting.'''

        from vlm.utils import (
            sample_vlm_system_prompt,
            get_vlm_system_prompt,
            sample_vlm_prompt_template,
            get_vlm_prompt_template
        )

        # Load default system prompt and template
        if sample_kwargs is None:
            self.system_prompt = get_vlm_system_prompt(model_name)
            self.prompt_template = get_vlm_prompt_template(model_name, prompt_type=prompt_type)

        # Otherwise, randomly sample; unless the seed is None
        else:
            if sample_kwargs['system_prompt_seed'] is None:
                self.system_prompt = get_vlm_system_prompt(model_name)
            else:
                self.system_prompt = sample_vlm_system_prompt(seed=sample_kwargs['system_prompt_seed'])

            if sample_kwargs['prompt_template_seed'] is None:
                self.prompt_template = get_vlm_prompt_template(model_name, prompt_type=prompt_type)
            else:
                self.prompt_template = sample_vlm_prompt_template(seed=sample_kwargs['prompt_template_seed'])

    def load_and_apply_prompt_template(
        self,
        model_name: str,
        sample_kwargs: Optional[Dict] = None,
        prompt_type: Optional[str] = 'zero-shot',
        system_prompt: Optional[str] = None,
        prompt_template: Optional[Union[Template,Tuple[Template]]] = None,
        llava_kwargs: Optional[Dict] = None
    ):
        '''Loads and applies the selected/sampled prompt template to the data.'''

        self.qas[self.main_split] = []

        # Load the prompt template
        if system_prompt is not None and prompt_template is not None:
            self.system_prompt = system_prompt
            self.prompt_template = prompt_template
        else:
            self.load_prompt_template(
                model_name, 
                self.name,
                prompt_type, 
                sample_kwargs=sample_kwargs
            )

        few_shot_qas = getattr(self, 'few_shot_qas', None)
        qas = self.qa_dict[self.main_split]
        pbar = tqdm(qas, desc=f'Applying template to "{self.main_split}" split', disable=(not self.verbose), unit='QA pair', total=len(qas))
        for (question, options, answer, image_input) in pbar:
            q_template = self.prompt_template['question']
            a_template = self.prompt_template['answer']

            # LLaVA models (specifically, for LLaVA-v0 and LLaVA-Med)
            if 'llava' in model_name:
                from llava.conversation import conv_templates
                from llava.constants import (
                    DEFAULT_IMAGE_TOKEN,
                    DEFAULT_IMAGE_PATCH_TOKEN,
                    DEFAULT_IM_START_TOKEN,
                    DEFAULT_IM_END_TOKEN
                )

                def preprocess_question_v1(question, n_images):
                    mm_use_im_start_end = llava_kwargs['mm_use_im_start_end']

                    if mm_use_im_start_end:
                        question = n_images * (
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
                        ) + question
                    else:
                        question = n_images * (DEFAULT_IMAGE_TOKEN + '\n') + question

                    return question
                
                def preprocess_question_v0(question, n_images):
                    mm_use_im_start_end = llava_kwargs['mm_use_im_start_end']
                    image_token_len = llava_kwargs['image_token_len']
                    
                    if mm_use_im_start_end:
                        question += n_images * (
                            '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
                        )
                    else:
                        question += n_images * ('\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len)

                    return question

                conv_mode = llava_kwargs['conv_mode']
                conv = conv_templates[conv_mode].copy()
                conv.system = '' if prompt_type == 'zero-shot-ft' else f'{self.system_prompt}\n'
                if 'v0' in conv_mode:
                    conv.sep = '\n### '
                
                preprocess_question = preprocess_question_v1 if 'llava-v1' in model_name else preprocess_question_v0

                if few_shot_qas is not None and prompt_type == 'few-shot':
                    for (fs_question, fs_options, fs_answer, fs_image_input) in few_shot_qas:
                        n_fs_images = len(re.findall(DEFAULT_IMAGE_TOKEN, fs_question))
                        fs_question = fs_question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                        fs_question = q_template.render(question=fs_question, options=fs_options)
                        fs_question = preprocess_question(fs_question, n_fs_images)
                        fs_answer = a_template.render(options=fs_options, answer=fs_answer)

                        conv.append_message(conv.roles[0], fs_question)
                        conv.append_message(conv.roles[1], fs_answer)
                    
                n_images = len(re.findall(DEFAULT_IMAGE_TOKEN, question))
                question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                question = q_template.render(question=question, options=options)
                question = preprocess_question(question, n_images)
                answer = a_template.render(options=options, answer=None)
                
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], answer) # NOTE: This still adds a separator, since answer is not None

                prompt = conv.get_prompt()
                if prompt.endswith(conv.sep):
                    prompt = prompt[:-len(conv.sep)]

            # Flamingo models
            elif 'flamingo' in model_name:
                prompt = '' if prompt_type == 'zero-shot-ft' else f'{self.system_prompt}\n'

                if few_shot_qas is not None and prompt_type == 'few-shot':
                    for (fs_question, fs_options, fs_answer, fs_image_input) in few_shot_qas:
                        fs_question = fs_question.replace('<image>', '').strip()
                        fs_question = q_template.render(question=fs_question, options=fs_options)
                        fs_answer = a_template.render(options=fs_options, answer=fs_answer)
                        fs_image_tags = '<image>' * len(fs_image_input)
                        fs_qa_pair = f'{fs_image_tags}{fs_question} {fs_answer}.<|endofchunk|>'
                        prompt += fs_qa_pair

                question = question.replace('<image>', '').strip()
                question = q_template.render(question=question, options=options)
                answer = a_template.render(options=options, answer=None)
                image_tags = '<image>' * len(image_input)
                qa_pair = f'{image_tags}{question} {answer}'
                prompt += qa_pair

            self.qas[self.main_split].append(prompt)

    def sample_few_shot_qas(self, n_shot: Optional[int] = 5, seed: Optional[int] = None) -> None:
        '''Randomly sample k QA pairs from the training set for few-shot prompting.'''

        if 'train' not in self.qa_dict:
            logging.error('Training set must be loaded before sampling few-shot examples.')
            raise RuntimeError
        
        # Randomly sample k QA pairs
        n_train = len(self.qa_dict['train'])
        few_shot_idxs = np.arange(n_train)
        generator = np.random.default_rng(seed if seed is not None else self.seed)
        permutation = generator.permutation(n_train)
        few_shot_idxs = few_shot_idxs[permutation][:n_shot]
        self.few_shot_qas = [self.qa_dict['train'][i] for i in few_shot_idxs]
        self.few_shot_images = [qa[3] for qa in self.few_shot_qas]

    def __len__(self) -> int:
        return len(self.qa_dict[self.main_split])
    
    # FIXME
    def __getitem__(self, idx: int) -> Tuple[str,str]:
        return self.qas[self.main_split][idx]

class VQARADDataset(VQADataset):
    '''VQA-RAD dataset.'''

    def __init__(
        self, 
        name: Optional[str] = 'vqa-rad',
        qa_dir: Optional[str] = osp.join(DATA_DIR, 'vqa-rad/closed'),
        image_dir: Optional[str] = osp.join(DATA_DIR, 'vqa-rad/images'),
        splits: Optional[List[str]] = ['test'],
        main_split: Optional[str] = 'test',
        seed: Optional[int] = 42,
        verbose: Optional[bool] = False,
        **kwargs
    ):
        super(VQARADDataset, self).__init__(
            name=name, 
            qa_dir=qa_dir, 
            image_dir=image_dir, 
            splits=splits,
            main_split=main_split,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

class PathVQADataset(VQADataset):
    '''PathVQA dataset.'''

    def __init__(
        self,
        name: Optional[str] = 'pvqa',
        qa_dir: Optional[str] = osp.join(DATA_DIR, 'pvqa/closed'),
        image_dir: Optional[str] = osp.join(DATA_DIR, 'pvqa/images'),
        splits: Optional[List[str]] = ['test'],
        main_split: Optional[str] = 'test',
        seed: Optional[int] = 42,
        verbose: Optional[bool] = False,
        **kwargs
    ):
        super(PathVQADataset, self).__init__(
            name=name, 
            qa_dir=qa_dir, 
            image_dir=image_dir, 
            splits=splits,
            main_split=main_split,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

class SlakeDataset(VQADataset):
    '''SLAKE dataset.'''

    def __init__(
        self,
        name: Optional[str] = 'slake',
        qa_dir: Optional[str] = osp.join(DATA_DIR, 'slake/closed'),
        image_dir: Optional[str] = osp.join(DATA_DIR, 'slake/imgs'),
        splits: Optional[List[str]] = ['test'],
        main_split: Optional[str] = 'test',
        seed: Optional[int] = 42,
        verbose: Optional[bool] = False,
        **kwargs
    ):
        super(SlakeDataset, self).__init__(
            name=name, 
            qa_dir=qa_dir, 
            image_dir=image_dir, 
            splits=splits,
            main_split=main_split,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

class MMMUDataset(VQADataset):
    '''
        MMMU dataset.

        Reference: https://mmmu-benchmark.github.io/

    '''

    def __init__(
        self,
        name: Optional[str] = 'mmmu_basic-medical-science',
        splits: Optional[List[str]] = ['test'],
        main_split: Optional[str] = 'test',
        seed: Optional[int] = 42,
        verbose: Optional[bool] = False,
        hf_cache_dir: Optional[str] = '/data/hf_models',
        **kwargs
    ):
        self.hf_cache_dir = hf_cache_dir
        
        super(MMMUDataset, self).__init__(
            name=name,
            qa_dir='',
            image_dir='',
            splits=splits,
            main_split=main_split,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

    def load_qas(self) -> None:
        '''Loads and preprocesses the given question-answer pairs.'''

        # Fetch HuggingFace dataset
        # e.g., 'mmmu_basic-medical-science' -> 'Basic_Medical_Science'
        self.subject_area = self.name.split('_')[-1].replace('-', '_').title().replace('And', 'and')
        hf_dataset = load_dataset('MMMU/MMMU', self.subject_area, cache_dir=self.hf_cache_dir)
        
        for split in self.splits:
            split_name_mapper = {'train': 'dev', 'val': 'validation', 'test': 'validation'}
            dataset = hf_dataset[split_name_mapper[split]]

            if split != 'train':
                dataset = dataset.train_test_split(train_size=5, seed=DEFAULT_SEED)
                dataset = dataset['train'] if split == 'val' else dataset['test']

            questions = []
            options = []
            answers = []
            images = []

            pbar = tqdm(dataset, desc=f'Loading the "{split}" split', unit='QA pair', disable=(not self.verbose), total=len(dataset))
            for sample in pbar:
                if sample['question_type'] == 'open':
                    continue

                question = sample['question']
                question = re.sub('<image \d+>', '<image>', question)
                question = self.replace_latex_symbols(question)
                question = question.strip()
                questions.append(question)

                qa_options = eval(sample['options'])
                qa_options = [self.replace_latex_symbols(o) for o in qa_options]
                options.append(qa_options)

                answer = options[-1][ord(sample['answer']) - ord('A')]
                answers.append(answer)
                
                sample_images = []
                for i in range(1,8):
                    if sample[f'image_{i}'] is not None:
                        sample_images.append(sample[f'image_{i}'])

                images.append(sample_images)
            
            self.qa_dict[split] = list(zip(questions, options, answers, images))

    def replace_latex_symbols(self, text):
        pattern = r'\$\\([^\$]+)\$'
        replacement = r' \1 '
        return re.sub(pattern, replacement, text).replace('  ', ' ').strip()