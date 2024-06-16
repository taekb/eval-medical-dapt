import os
import re
import numpy as np
from tqdm import tqdm
import random
import string
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

from jinja2 import Template

DEFAULT_SEED = 42
MODELS_REQ_LOGIN = [
    'meta-llama/Llama-2-7b-hf',
    'epfl-llm/meditron-7b',
    'mistralai/Mistral-7B-Instruct-v0.1'
]
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
# Llama-3 Chat Template
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
# Mistral Chat Template
# NOTE: Same as original, just added a space after [/INST]
MISTRAL_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST] ' }}" # Here
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token + ' ' }}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)

'''
    Utility functions for preprocessing/normalizing strings.

    normalize_text() adapted from the HELM repository.
    Reference: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/metrics/evaluate_reference_metrics.py#L41

'''

def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)

def white_space_fix(text: str) -> str:
    return " ".join(text.split())

def remove_punc(text: str) -> str:
    '''Modified from the original function.'''
    exclude = set(string.punctuation)
    #return "".join(ch for ch in text if ch not in exclude)

    for ch in exclude:
        text = text.replace(ch, ' ')

    return text

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    return white_space_fix(remove_articles(remove_punc(text.lower())))

'''
    Utility functions for tokenization.

'''

def allows_system_prompts(tokenizer):
    '''Checks if the chat template for a tokenizer allows system prompts.'''

    allows = False
    dummy_conv = [
        {'role': 'system', 'content': 'System prompt.'},
        {'role': 'user', 'content': 'User prompt.'},
        {'role': 'assistant', 'content': 'Assistant prompt.'}
    ]
    try:
        tokenizer.apply_chat_template(dummy_conv, tokenize=False)
        allows = True
    except:
        allows = False
    
    return allows

'''
    Utilities for prompt templates.

    Prompt template generation approach is adapted from Sclar et al. (2024).

'''

# System prompt pool
CHOSEN_SYS_PROMPT_LIST = [
    'The following is a multiple-choice question about medical knowledge. Answer the question by choosing one of the options from A to {}.',
    'This is a multiple-choice medical knowledge question. Select your answer from options A to {}.',
    'Answer the following multiple-choice question on medical knowledge by choosing from A to {}.',
    'Choose an answer for the medical knowledge multiple-choice question from A to {}.',
    'Below is a multiple-choice query about medical knowledge. Please select from A to {}.',
    'The next item is a medical knowledge multiple-choice question. Pick one option from A to {}.',
    'For the following medical knowledge multiple-choice question, select your answer from A to {}.',
    'Respond to this multiple-choice question on medical knowledge by selecting from A to {}.',
    'This multiple-choice question covers medical knowledge. Choose your answer from A to {}.',
    'Select your answer to the upcoming multiple-choice question on medical knowledge from A to {}.',
    'Answer the multiple-choice question below on medical knowledge by choosing from A to {}.'
]

# Grammar definition
CHOSEN_SEPARATOR_LIST = [
    ': ', ' : ', ' :: ', ':\n',
    '= ', ' = ', ' == ', '=\n',
    ' - ', ' -- ', ' --- ',
    '\n', '\n\n'
]
CHOSEN_SPACE_LIST = ['\n', '\n\n', ' || ', ' ']
CHOSEN_OPTION_SPACE_LIST = ['\n', '\n\n', '; ', ';\n', ';\n\n', ' || ', ' ', ', ']
Q_DESC_LIST = ['', 'Question']
O_DESC_LIST = ['', 'Options', 'Choices']
A_DESC_LIST = ['Answer', 'The answer is']

TEXT_DESCRIPTOR_FN_LIST = [ 
    (lambda x: x, "lambda x: x"),
    (lambda x: x.title(), "lambda x: x.title()"),
    (lambda x: x.upper(), "lambda x: x.upper()"),
    (lambda x: x.lower(), "lambda x: x.lower()"),
    (lambda x: f'### {x}', "lambda x: f'### {x}"),
    (lambda x: f'**{x}**', "lambda x: f'**{x}**")
]
ITEM_WRAPPER_LIST = [ # F_wrapper 
    (lambda x: f'({x})', "lambda x: f'({x})'"),
    (lambda x: f'{x}.', "lambda x: f'{x}.'"),
    (lambda x: f'{x})', "lambda x: f'{x})'"),
    (lambda x: f'{x} )', "lambda x: f'{x} )'"),
    (lambda x: f'[{x}]', "lambda x: f'[{x}]'"),
    (lambda x: f'<{x}>', "lambda x: f'<{x}>'"),
]
NUMBER_FORMAT_LIST = [(lambda x: chr(ord('A') + x), "lambda x: chr(ord('A') + x)")]

def sample_llm_system_prompt(seed=42):
    '''Randomly samples a system prompt from system prompt pool defined above.'''

    random.seed(seed)
    system_prompt = random.choice(CHOSEN_SYS_PROMPT_LIST)

    return system_prompt

def sample_llm_prompt_template(seed=42):
    '''Randomly samples a jinja template according to the grammar defined above.'''

    random.seed(seed)
    space = random.choice(CHOSEN_SPACE_LIST)
    
    # Add constraint as in Sclar et al. (2024).
    if '\n' not in space:
        sep = random.choice([s for s in CHOSEN_SEPARATOR_LIST if '\n' not in s])
        o_space = random.choice([s for s in CHOSEN_OPTION_SPACE_LIST if '\n' not in s])
    else:
        sep = random.choice(CHOSEN_SEPARATOR_LIST)
        o_space = random.choice(CHOSEN_OPTION_SPACE_LIST)

    f_case = random.choice(TEXT_DESCRIPTOR_FN_LIST)[0]

    # Define question template
    q_template_str = ''
    q_desc = random.choice(Q_DESC_LIST)
    empty_q_desc = (q_desc == '')
    q_desc = f_case(q_desc) if not empty_q_desc else ''
    q_sep = sep if not empty_q_desc else ''
    q_template_str += "{% " + f"set q_str = '{q_desc}{q_sep}' + question + '{space}'" + " %}"
    q_template_str += "{{ q_str }}"
    
    o_desc = random.choice(O_DESC_LIST)
    empty_o_desc = (o_desc == '')
    o_sep = sep if not empty_o_desc else ''
    o_desc = f_case(o_desc) if not empty_o_desc else ''
    f_num = random.choice(NUMBER_FORMAT_LIST)[0]
    f_wrapper = random.choice(ITEM_WRAPPER_LIST)[0]
    def func_num(x): return f_num(x)
    def func_wrapper(x): return f_wrapper(x)
    
    q_template_str += "{% " + f"set o_str = '{o_desc}{o_sep}'" + "%}"
    q_template_str += "{{ o_str }}"
    q_template_str += "{% for option in options %}"
    q_template_str += "{{ func_wrapper(func_num(loop.index0)) + ' ' + option }}"
    q_template_str += "{% if not loop.last %}" + f"{o_space}" + "{% endif %}"
    q_template_str += "{% endfor %}"
            
    q_template = Template(q_template_str)
    q_template.globals.update(dict(func_num=func_num, func_wrapper=func_wrapper))

    # Define answer template
    a_template_str = ''
    a_desc = random.choice(A_DESC_LIST)
    a_sep = sep if a_desc != '' else ''
    a_desc = f_case(a_desc)
    def get_a_idx(options, answer): return options.index(answer)
    
    a_template_str += "{% if answer is not none and answer != '' %}"
    a_template_str += (
        "{% " 
        f"set answer_str = '{a_desc}{a_sep}' + func_wrapper(func_num(get_a_idx(options, answer))) + ' ' + answer"
        " %}"
    )
    a_template_str += "{{ answer_str }}"
    a_template_str += "{% else %}"
    a_template_str += "{{ " + f"'{a_desc}{a_sep}'" + " }}"
    a_template_str += "{% endif %}"
    a_template = Template(a_template_str)
    a_template.globals.update(dict(func_num=func_num, func_wrapper=func_wrapper, get_a_idx=get_a_idx))    

    # Define full template
    template_str = (
        "{% if system_prompt is not none and system_prompt != '' %}"
        "{{ system_prompt }}"
        "{{ " + f"'{space}'" + " }}"
        "{% endif %}"
        "{% if few_shot_qas is not none %}"
        "{% for few_shot_qa in few_shot_qas %}"
        "{% set question = few_shot_qa[0] %}"
        "{% set options = few_shot_qa[1] %}"
        "{% set answer = few_shot_qa[2] %}"
    )
    template_str += q_template_str + "{{ " + f"'{space}'" + " }}"
    template_str += a_template_str + "{{ " + f"'{space}'" + " }}"
    template_str += (
        "{% endfor %}"
        "{% endif %}"
        "{% set question = qa[0] %}"
        "{% set options = qa[1] %}"
        "{% set answer = qa[2] %}"
    )
    template_str += q_template_str + "{{ " + f"'{space}'" + " }}"
    template_str += a_template_str
    template = Template(template_str)
    template.globals.update(dict(func_num=func_num, func_wrapper=func_wrapper, get_a_idx=get_a_idx))

    template_dict = dict(
        full=template,
        question=q_template,
        answer=a_template,
        template_str=dict(
            full=template_str,
            question=q_template_str,
            answer=a_template_str,
            q_header=f'{q_desc}{q_sep}',
            o_header=f'{o_desc}{o_sep}',
            a_header=f'{a_desc}{a_sep}'
        ),
        f_num=f_num,
        f_wrapper=f_wrapper,
        get_a_idx=get_a_idx
    )

    return template_dict

'''
    Default system prompts and templates used by the authors of each model.

'''

def get_llm_system_prompt(model_name, dataset_name):
    '''Returns the default system prompt for the specified model.'''

    # MediTron (Chen et al., 2023)
    if 'meditron' in model_name:
        if dataset_name == 'medqa':
            system_prompt = (
                'You are a medical doctor taking the US Medical Licensing Examination. '
                'You need to demonstrate your understanding of basic and clinical science, '
                'medical knowledge, and mechanisms underlying health, disease, patient care, '
                'and modes of therapy. Show your ability to apply the knowledge essential '
                'for medical practice. For the following multiple-choice question, select one '
                f'correct answer from A to E. Base your answer on '
                'the current and standard practices referenced in medical guidelines.'
            )
        
        elif dataset_name == 'medmcqa' or 'mmlu' in dataset_name:
            system_prompt = (
                'You are a medical doctor answering real-world medical entrance exam questions. ' 
                'Based on your understanding of basic and clinical science, medical knowledge, and '
                'mechanisms underlying health, disease, patient care, and modes of therapy, '
                f'answer the following multiple-choice question. Select one correct answer from A to D. '
                'Base your answer on the current and standard practices referenced in medical guidelines.'
            )
        
        elif dataset_name == 'pubmedqa':
            system_prompt = (
                'As an expert doctor in clinical science and medical knowledge, can you tell me '
                'if the following statement is correct? Answer yes, no, or maybe.'
            )

    # BioMistral
    elif 'biomistral' in model_name:
        system_prompt = 'The following are multiple choice questions (with answers) about medical knowledge.'

    # OpenBioLLM
    elif 'openbiollm' in model_name:
        system_prompt = (
            "You are an expert and experienced from the healthcare and biomedical domain "
            "with extensive medical knowledge and practical experience. "
            "Your name is OpenBioLLM, and you were developed by Saama AI Labs. "
            "who's willing to help answer the user's query with explanation. "
            "In your explanation, leverage your deep medical expertise such as "
            "relevant anatomical structures, physiological processes, diagnostic criteria, "
            "treatment guidelines, or other pertinent medical concepts. "
            "Use precise medical terminology while still aiming to make the explanation clear "
            "and accessible to a general audience."
        )

    else:
        n_mcq = N_MCQ[dataset_name]
        system_prompt = (
            'The following is a multiple-choice question about medical knowledge. '
            f'Answer the question by choosing one of the options from A to {chr(ord("A")+(n_mcq-1))}.'
        )

    return system_prompt

def get_llm_prompt_template(model_name, prompt_type='zero-shot'):
    '''Returns the default jinja template for the specified model.'''

    # MediTron (Chen et al., 2023)
    if 'meditron' in model_name:
        # Fine-tuning prompt template
        # NOTE: Does not support few-shot examples in current form
        if prompt_type == 'zero-shot-ft':
            def f_num(x): return chr(ord('A') + x)
            def f_wrapper(x): return f'({x})'
            def get_a_idx(options, answer): return options.index(answer)

            # Full template
            template_str = (
                "{% if system_prompt is not none and system_prompt != '' %}"
                "{{ '<|im_start|> system\n' }}"
                "{{ system_prompt + '<|im_end|>\n' }}"
                "{% endif %}"
                "{{ '<|im_start|> question\n' }}"
                "{{ 'Question: ' + qa[0] + '\n' }}"
                "{{ 'Options:\n' }}"
                "{% set options = qa[1] %}"
                "{% for option in options %}"
                "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
                "{{ option_str }}"
                "{% if loop.last %}{{ '<|im_end|>\n' }}"
                "{% else %}{{ '\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{{ '<|im_start|> answer\n' }}"
                "{% set answer = qa[2] %}"
                "{% if answer is not none and answer != '' %}"
                "{% set answer_str = f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer %}"
                "{{ answer_str }}"
                "{% endif %}"
            )
            template = Template(template_str)
            template.globals.update(dict(f_num=f_num, f_wrapper=f_wrapper, get_a_idx=get_a_idx))

            # Question template
            q_template_str = (
                "{{ '<|im_start|> question\n' }}"
                "{{ 'Question: ' + question + '\n' }}"
                "{{ 'Options:\n' }}"
                "{% for option in options %}"
                "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
                "{{ option_str }}"
                "{% if loop.last %}{{ '<|im_end|>' }}"
                "{% else %}{{ '\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )
            q_header = ['<|im_start|>', 'Question:']
            o_header = 'Options:'

            # Answer template
            a_template_str = (
                "{{ '<|im_start|> answer\n' }}"
                "{% if answer is not none and answer != '' %}"
                "{{ f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer }}"
                "{% endif %}"
            )
            a_header = '<|im_start|>'
        
        else:
            def f_num(x): return chr(ord('A') + x)
            def f_wrapper(x): return f'{x}.'
            def get_a_idx(options, answer): return options.index(answer)

            # Full template
            template_str = (
                "{% if system_prompt is not none and system_prompt != '' %}"
                "{{ system_prompt + '\n' }}"
                "{% endif %}"
                "{% if few_shot_qas is not none %}"
                "{% for few_shot_qa in few_shot_qas %}"
                "{% set demo_question = 'Question: ' + few_shot_qa[0] + '\n\n' %}"
                "{{ demo_question }}"
                "{{ 'Options:\n' }}"
                "{% set demo_options = few_shot_qa[1] %}"
                "{% for demo_option in demo_options %}"
                "{% set demo_option_str = f_wrapper(f_num(loop.index0)) + ' ' + demo_option + '\n' %}"
                "{{ demo_option_str }}"
                "{% endfor %}"
                "{% set demo_answer_str = 'The answer is: ' + f_num(get_a_idx(demo_options, few_shot_qa[2])) + '\n\n' %}"
                "{{ demo_answer_str }}"
                "{% endfor %}"
                "{% endif %}"
                "{% set question = 'Question: ' + qa[0] + '\n\n' %}"
                "{{ question }}"
                "{{ 'Options:\n' }}"
                "{% set options = qa[1] %}"
                "{% for option in options %}"
                "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option + '\n' %}"
                "{{ option_str }}"
                "{% endfor %}"
                "{{ 'The answer is: ' }}"
                "{% set answer = qa[2] %}"
                "{% if answer is not none and answer != '' %}"
                "{% set answer_str = f_num(get_a_idx(options, answer)) %}"
                "{{ answer_str }}"
                "{% endif %}"
            )

            # Question template
            q_template_str = (
                "{% set question_str = 'Question: ' + question + '\n\n' %}"
                "{{ question_str }}"
                "{{ 'Options:\n' }}"
                "{% for option in options %}"
                "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
                "{{ option_str }}"
                "{% if not loop.last %}"
                "{{ '\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )
            q_header = 'Question:'
            o_header = 'Options:'
            
            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{% set answer_str = 'The answer is: ' + f_num(get_a_idx(options, answer)) %}"
                "{% else %}"
                "{% set answer_str = 'The answer is: ' %}"
                "{% endif %}"
                "{{ answer_str }}"
            )
            a_header = 'The answer is:'

    # Prompt template for BioMedGPT-LM (Luo et al., 2023)
    elif 'biomedgpt' in model_name:
        def f_num(x): return chr(ord('A') + x)
        def f_wrapper(x): return f'({x})'
        def get_a_idx(options, answer): return options.index(answer)

        # Full template
        template_str = (
            "{% if system_prompt is not none and system_prompt != '' %}"
            "{{ system_prompt + ' ' }}"
            "{% endif %}"
            "{% if few_shot_qas is not none %}"
            "{% for few_shot_qa in few_shot_qas %}"
            "{% set demo_question = '### Human: ' + few_shot_qa[0] + ' ' %}"
            "{{ demo_question }}"
            "{% set demo_options = few_shot_qa[1] %}"
            "{% for demo_option in demo_options %}"
            "{% set demo_option_str = f_wrapper(f_num(loop.index0)) + ' ' + demo_option + ' ' %}"
            "{{ demo_option_str }}"
            "{% endfor %}"
            "{{ '### Assistant: ' }}"
            "{% set demo_answer_str = f_wrapper(f_num(get_a_idx(demo_options, few_shot_qa[2]))) + ' ' + few_shot_qa[2] + ' ' %}"
            "{{ demo_answer_str }}"
            "{% endfor %}"
            "{% endif %}"
            "{% set question = '### Human: ' + qa[0] + ' ' %}"
            "{{ question }}"
            "{% set options = qa[1] %}"
            "{% for option in options %}"
            "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option + ' ' %}"
            "{{ option_str }}"
            "{% endfor %}"
            "{{ '### Assistant: ' }}"
            "{% set answer = qa[2] %}"
            "{% if answer is not none and answer != '' %}"
            "{% set answer_str = f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer %}"
            "{{ answer_str }}"
            "{% endif %}"
        )

        # Question template
        q_template_str = (
            "{{ '### Human: ' + question + ' ' }}"
            "{% for option in options %}"
            "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
            "{{ option_str }}"
            "{% if not loop.last %}"
            "{{ ' ' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        q_header = '### Human:'
        o_header = ''

        # Answer template
        a_template_str = (
            "{{ '### Assistant: ' }}"
            "{% if answer is not none and answer != '' %}"
            "{{ f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer  }}"
            "{% endif %}"
        )
        a_header = '### Assistant:'
    
    # Prompt template for BioMistral (Labrak et al., 2024)
    elif 'biomistral' in model_name:
        def f_num(x): return chr(ord('A') + x)
        def f_wrapper(x): return f'({x})'
        def get_a_idx(options, answer): return options.index(answer)

        # Full template
        template_str = (
            "{% if system_prompt is not none and system_prompt != '' %}"
            "{{ system_prompt + '\n' }}"
            "{% endif %}"
            "{% if few_shot_qas is not none %}"
            "{% for few_shot_qa in few_shot_qas %}"
            "{% set demo_question = '**Question:** ' + few_shot_qa[0] + '\n' %}"
            "{{ demo_question }}"
            "{% set demo_options = few_shot_qa[1] %}"
            "{% for demo_option in demo_options %}"
            "{% set demo_option_str = f_wrapper(f_num(loop.index0)) + ' ' + demo_option + '\n' %}"
            "{{ demo_option_str }}"
            "{% endfor %}"
            "{% set demo_answer_str = '**Answer:** (' + f_num(get_a_idx(demo_options, few_shot_qa[2])) + '\n\n' %}"
            "{{ demo_answer_str }}"
            "{% endfor %}"
            "{% endif %}"
            "{% set question = '**Question:** ' + qa[0] + '\n' %}"
            "{{ question }}"
            "{% set options = qa[1] %}"
            "{% for option in options %}"
            "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option + '\n' %}"
            "{{ option_str }}"
            "{% endfor %}"
            "{{ '**Answer:** (' }}"
            "{% set answer = qa[2] %}"
            "{% if answer is not none and answer != '' %}"
            "{% set answer_str = f_num(get_a_idx(options, answer)) %}"
            "{{ answer_str }}"
            "{% endif %}"
        )

        # Question template
        q_template_str = (
            "{{ '**Question:** ' + question + '\n' }}"
            "{% for option in options %}"
            "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
            "{{ option_str }}"
            "{% if not loop.last %}"
            "{{ '\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        q_header = '**Question:**'
        o_header = ''

        # Answer template
        a_template_str = (
            "{{ '**Answer:** (' }}"
            "{% if answer is not none and answer != '' %}"
            "{{ f_num(get_a_idx(options, answer)) }}"
            "{% endif %}"
        )
        a_header = '**Answer:**'

    elif any([m in model_name for m in CHAT_MODELS]):
        def f_num(x): return chr(ord('A') + x)
        def f_wrapper(x): return f'({x})'
        def get_a_idx(options, answer): return options.index(answer)

        # Full template
        # NOTE: Full template is not defined for chat models
        template_str = None

        # Question template
        q_template_str = (
            "{{ 'Question: ' + question + '\n' }}"
            "{% for option in options %}"
            "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
            "{{ option_str }}"
            "{% if not loop.last %}"
            "{{ '\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        q_header = 'Question:'
        o_header = ''
        
        # Answer template
        a_template_str = (
            "{% if answer is not none and answer != '' %}"
            "{% set answer_str = f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer %}"
            "{% else %}"
            "{% set answer_str = '' %}"
            "{% endif %}"
            "{{ answer_str }}"
        )
        a_header = ''

    # Default template for all other models
    else:
        def f_num(x): return chr(ord('A') + x)
        def f_wrapper(x): return f'({x})'
        def get_a_idx(options, answer): return options.index(answer)

        # Full template
        template_str = (
            "{% if system_prompt is not none and system_prompt != '' %}"
            "{{ system_prompt + '\n\n' }}"
            "{% endif %}"
            "{% if few_shot_qas is not none %}"
            "{% for few_shot_qa in few_shot_qas %}"
            "{% set demo_question = '### Question: ' + few_shot_qa[0] + '\n' %}"
            "{{ demo_question }}"
            "{% set demo_options = few_shot_qa[1] %}"
            "{% for demo_option in demo_options %}"
            "{% set demo_option_str = f_wrapper(f_num(loop.index0)) + ' ' + demo_option + '\n' %}"
            "{{ demo_option_str }}"
            "{% endfor %}"
            "{{ '### Answer: ' }}"
            "{% set demo_answer_str = f_wrapper(f_num(get_a_idx(demo_options, few_shot_qa[2]))) + ' ' + few_shot_qa[2] + '\n\n' %}"
            "{{ demo_answer_str }}"
            "{% endfor %}"
            "{% endif %}"
            "{% set question = '### Question: ' + qa[0] + '\n' %}"
            "{{ question }}"
            "{% set options = qa[1] %}"
            "{% for option in options %}"
            "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option + '\n' %}"
            "{{ option_str }}"
            "{% endfor %}"
            "{{ '### Answer: ' }}"
            "{% set answer = qa[2] %}"
            "{% if answer is not none and answer != '' %}"
            "{% set answer_str = f_wrapper(f_num(get_a_idx(qa[1], answer))) + ' ' + answer %}"
            "{{ answer_str }}"
            "{% endif %}"
        )

        # Question template
        q_template_str = (
            "{% set question_str = '### Question: ' + question + '\n' %}"
            "{{ question_str }}"
            "{% for option in options %}"
            "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
            "{{ option_str }}"
            "{% if not loop.last %}"
            "{{ '\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        q_header = '### Question:'
        o_header = ''
        
        # Answer template
        a_template_str = (
            "{% if answer is not none and answer != '' %}"
            "{% set answer_str = '### Answer: ' + f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer %}"
            "{% else %}"
            "{% set answer_str = '### Answer: ' %}"
            "{% endif %}"
            "{{ answer_str }}"
        )
        a_header = '### Answer:'
    
    f_dict = dict(f_num=f_num, f_wrapper=f_wrapper, get_a_idx=get_a_idx)
    
    try:
        template = Template(template_str)
        template.globals.update(f_dict)
    except:
        logging.info('Full template not available for chat models. Skipping templatization.')
        template = None

    q_template = Template(q_template_str)
    a_template = Template(a_template_str)
    q_template.globals.update(f_dict)
    a_template.globals.update(f_dict)

    template_dict = dict(
        full=template,
        question=q_template,
        answer=a_template,
        template_str=dict(
            full=template_str,
            question=q_template_str,
            answer=a_template_str,
            q_header=q_header,
            o_header=o_header,
            a_header=a_header
        ),
        f_num=f_num,
        f_wrapper=f_wrapper,
        get_a_idx=get_a_idx
    )

    return template_dict