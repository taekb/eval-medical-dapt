import re
import string
import random
from jinja2 import Template

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s | %(message)s]',
    datefmt='%d-%b-%y %H:%M:%S'
)

DEFAULT_SEED = 42

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

    for ch in exclude:
        text = text.replace(ch, ' ')

    return text

def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    return white_space_fix(remove_articles(remove_punc(text.lower())))

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

def sample_vlm_system_prompt(seed=42):
    '''Randomly samples a system prompt from system prompt pool defined above.'''

    random.seed(seed)
    system_prompt = random.choice(CHOSEN_SYS_PROMPT_LIST)

    return system_prompt

def sample_vlm_prompt_template(seed=42):
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

    template_dict = dict(
        question=q_template,
        answer=a_template,
        template_str=dict(
            question=q_template_str,
            answer=a_template_str
        ),
        f_num=f_num,
        f_wrapper=f_wrapper,
        get_a_idx=get_a_idx
    )

    return template_dict


'''
    Default system prompts and templates used by the authors of each model.

'''

def get_vlm_system_prompt(model_name):
    '''Returns the default system prompt for the specified model.'''

    # Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py#L301
    if 'llava-v0' in model_name:
        system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions."
        )
    
    # Reference: https://github.com/microsoft/LLaVA-Med/blob/b9a98a736d2ef05bcf5ff345be6403fb3a664eaf/llava/conversation.py#L270
    elif 'llava-med' in model_name:
        system_prompt = (
            "You are LLaVA-Med, a large language and vision assistant trained by a group of researchers at Microsoft, "
            "based on the general domain LLaVA architecture. You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of medical and clinical tasks using natural language."
            "Follow the instructions carefully and explain your answers in detail."
        )

    elif 'med-flamingo' in model_name:
        system_prompt = (
            "You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. "
            "Follow the examples and answer the last question."
        )
    
    else:
        system_prompt = (
            'The following is a multiple-choice visual question requiring medical knowledge. '
            f'Answer the question by choosing one of the provided answer options.'
        )

    return system_prompt

def get_vlm_prompt_template(model_name, prompt_type='zero-shot'):
    '''Returns the default jinja template for the specified model.'''

    if 'llava-v0' in model_name:
        # Fine-tuning prompt template
        if prompt_type == 'zero-shot-ft':
            f_num = None
            f_wrapper = None
            get_a_idx = None

            # Question template
            q_template_str = "{{ question }}"

            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ answer }}"
                "{% endif %}"
            )
        
        else:
            def f_num(x): return chr(ord('A') + x)
            def f_wrapper(x): return f'({x})'
            def get_a_idx(options, answer): return options.index(answer)

            # Question template
            q_template_str = (
                "{% set question_str = question + '\n' %}"
                "{{ question_str }}"
                "{{ 'Options: ' }}"
                "{% for option in options %}"
                "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
                "{{ option_str }}"
                "{% if not loop.last %}"
                "{{ ' ' }}"
                "{% endif %}"
                "{% endfor %}"
            )

            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer }}"
                "{% endif %}"
            )

    # Reference: https://github.com/microsoft/LLaVA-Med/blob/b9a98a736d2ef05bcf5ff345be6403fb3a664eaf/llava/eval/eval_metrics/answer-file-llava-zeorshot.jsonl
    elif 'llava-med' in model_name:
        # Fine-tuning prompt template
        if prompt_type == 'zero-shot-ft':
            f_num = None
            f_wrapper = None
            get_a_idx = None

            # Question template
            q_template_str = "{{ question }}"

            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ answer }}"
                "{% endif %}"
            )
        
        else:
            def f_num(x): return chr(ord('A') + x)
            def f_wrapper(x): return f'({x})'
            def get_a_idx(options, answer): return options.index(answer)

            # Question template
            q_template_str = (
                "{% set question_str = question + ' ' %}"
                "{{ question_str }}"
                "{{ 'Please choose from the following options: [' }}"
                "{% for option in options %}"
                "{{ option }}"
                "{% if not loop.last %}"
                "{{ ', ' }}"
                "{% endif %}"
                "{% endfor %}"
                "{{ '].' }}"
            )

            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ answer }}"
                "{% endif %}"
            )
        
    elif 'open-flamingo' in model_name:
        # Fine-tuning prompt template
        if prompt_type == 'zero-shot-ft':
            f_num = None
            f_wrapper = None
            get_a_idx = None

            # Question template
            q_template_str = "{{ question }}"
            
            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ answer }}"
                "{% endif %}"
            )

        else:
            def f_num(x): return chr(ord('A') + x)
            def f_wrapper(x): return f'({x})'
            def get_a_idx(options, answer): return options.index(answer)

            # Question template
            q_template_str = (
                "{% set question_str = 'Question: ' + question + ' ' %}"
                "{{ question_str }}"
                "{% for option in options %}"
                "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
                "{{ option_str }}"
                "{% if not loop.last %}"
                "{{ ', ' }}"
                "{% else %}"
                "{{ ';' }}"
                "{% endif %}"
                "{% endfor%}"
            )

            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ 'Short answer: ' + f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer }}"
                "{% else %}"
                "{{ 'Short answer: ' }}"
                "{% endif %}"
            )

    elif 'med-flamingo' in model_name:
        # Fine-tuning prompt template
        if prompt_type == 'zero-shot-ft':
            f_num = None
            f_wrapper = None
            get_a_idx = None

            # Question template
            q_template_str = "{{ question }}"
            
            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ answer }}"
                "{% endif %}"
            )

        else:
            def f_num(x): return chr(ord('A') + x)
            def f_wrapper(x): return f'({x})'
            def get_a_idx(options, answer): return options.index(answer)

            # Question template
            q_template_str = (
                "{% set question_str = 'Question: ' + question + ' ' %}"
                "{{ question_str }}"
                "{% for option in options %}"
                "{% set option_str = f_wrapper(f_num(loop.index0)) + ' ' + option %}"
                "{{ option_str }}"
                "{% if not loop.last %}"
                "{{ ', ' }}"
                "{% else %}"
                "{{ ';' }}"
                "{% endif %}"
                "{% endfor%}"
            )

            # Answer template
            a_template_str = (
                "{% if answer is not none and answer != '' %}"
                "{{ 'Answer: ' + f_wrapper(f_num(get_a_idx(options, answer))) + ' ' + answer }}"
                "{% else %}"
                "{{ 'Answer: ' }}"
                "{% endif %}"
            )

    else:
        logging.error(f'Template not defined for model "{model_name}"')
        raise NotImplementedError
        
    f_dict = dict(f_num=f_num, f_wrapper=f_wrapper, get_a_idx=get_a_idx)

    q_template = Template(q_template_str)
    a_template = Template(a_template_str)
    q_template.globals.update(f_dict)
    a_template.globals.update(f_dict)

    template_dict = dict(
        question=q_template,
        answer=a_template,
        template_str=dict(question=q_template_str, answer=a_template_str),
        f_num=f_num,
        f_wrapper=f_wrapper,
        get_a_idx=get_a_idx
    )

    return template_dict