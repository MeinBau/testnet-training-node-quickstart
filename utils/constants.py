qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n",
    "assistant_format": "{content}<|im_end|>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|im_start|>tool\n{content}<|im_end|>\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<eos>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<start_of_turn>tool\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "system": None,
}

model2template = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen_template",
    "Qwen/Qwen2.5-3B-Instruct": "qwen_template",
    "Qwen/Qwen1.5-0.5B": "qwen_template",
    "Qwen/Qwen1.5-1.8B": "qwen_template",
    "Qwen/Qwen1.5-7B": "qwen_template",
}

model2size = {
    "Qwen/Qwen2.5-7B-Instruct": 7_620_000_000,
    "Qwen/Qwen2.5-3B-Instruct": 3_090_000_000,
    "Qwen/Qwen1.5-0.5B": 620_000_000,
    "Qwen/Qwen1.5-1.8B": 1_840_000_000,
    "Qwen/Qwen1.5-7B": 7_720_000_000,
}

model2base_model = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen1.5", # just use qwen1.5 here, it's the same model family
    "Qwen/Qwen2.5-3B-Instruct": "qwen1.5",
    "Qwen/Qwen1.5-0.5B": "qwen1.5",
    "Qwen/Qwen1.5-1.8B": "qwen1.5",
    "Qwen/Qwen1.5-7B": "qwen1.5",
}
