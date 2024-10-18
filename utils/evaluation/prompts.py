"""modified from fastchat.conversation"""


def instagger_prompt(input_string):
    return (
        "You are a helpful assistant. "
        "Please identify tags of user intentions in the following user query and provide an explanation for each tag. "
        'Please respond in the JSON format {"tag": str, "explanation": str}. '
        f'Query: "{input_string}" '
        "Assistant: "
    )


def llama2_prompt(input_string):
    return f"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n{input_string} [/INST]"


def vicuna_prompt(input_string):
    return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input_string} ASSISTANT:"


def openchat_prompt(input_string):
    return f"GPT4 Correct User: {input_string}<|end_of_turn|>GPT4 Correct Assistant:"


def wizardlm_prompt(input_string):
    return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input_string} ASSISTANT:"


def wizardcoder_prompt(input_string):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input_string}\n\n### Response:"


def wizardmath_prompt(input_string):
    # ref: https://huggingface.co/WizardLM/WizardMath-13B-V1.0
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input_string}\n\n### Response:"


def phi_2_coder_prompt(input_string):
    # ref: https://huggingface.co/mrm8488/phi-2-coder
    return f"Instruct: {input_string}\nOutput:"


def dolphin_2_6_phi_2_prompt(input_string):
    # ref: https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2
    return f"<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{input_string}<|im_end|>\n<|im_start|>assistant"


def phi_2_dpo_prompt(input_string):
    # ref: https://huggingface.co/lxuechen/phi-2-dpo
    return f"### Human: {input_string}\n\n### Assistant:"


def phi_2_sft_dpo_gpt4_en_ep1_prompt(input_string):
    # ref: None: https://huggingface.co/Yhyu13/phi-2-sft-alpaca_gpt4_en-ep1-lora
    return f"{input_string}"


def chatml_prompt(input_string):
    return f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{input_string}<|im_end|>\n<|im_start|>assistant"


def get_prompted_inputs(input_string, model_name):
    # if "llama" in model_name.lower():
    #     return llama2_prompt(input_string)
    # elif "vicuna" in model_name.lower():
    #     return vicuna_prompt(input_string)
    # elif "openchat" in model_name.lower():
    #     return openchat_prompt(input_string)
    # elif "wizard" in model_name.lower() and "lm" in model_name.lower():
    #     return wizardlm_prompt(input_string)
    # elif "wizard" in model_name.lower() and "coder" in model_name.lower():
    #     return wizardcoder_prompt(input_string)
    # elif "wizard" in model_name.lower() and "math" in model_name.lower():
    #     return wizardmath_prompt(input_string)
    # elif "phi-2-coder" in model_name.lower():
    #     return phi_2_coder_prompt(input_string)
    # elif "phi-2-dpo" in model_name.lower():
    #     return phi_2_dpo_prompt(input_string)
    # elif "dolphin-2_6-phi-2" in model_name.lower():
    #     return dolphin_2_6_phi_2_prompt(input_string)
    # elif "phi-2-sft-dpo-gpt4_en-ep1" in model_name.lower():
    #     return phi_2_sft_dpo_gpt4_en_ep1_prompt(input_string)
    # else:
    #     return input_string
    return input_string