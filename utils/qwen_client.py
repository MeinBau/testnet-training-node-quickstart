from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

sub_model_id = "Qwen/Qwen2.5-0.5B"  
device = "cuda" if torch.cuda.is_available() else "cpu"

sub_tokenizer = AutoTokenizer.from_pretrained(
        sub_model_id,
        use_fast=True,
    )
sub_model = AutoModelForCausalLM.from_pretrained(
    sub_model_id,
    device_map={"": 0},
    token=os.environ["HF_TOKEN"],
)

def ask_qwen_local(user_input: str) -> str:
    """
    Determines whether an API call is needed using the local Qwen model.
    """
    prompt = f"""User question: "{user_input}"
1. If the question contains a specific stock symbol (e.g., AAPL, TSLA), respond in the format 'CALL_API: SYMBOL'.
2. Otherwise, provide a plain text answer.
"""



    inputs = sub_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = sub_model.generate(**inputs, max_length=256, temperature=0)
    text = sub_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt and extract only the model's actual response
    response = text.split("User question")[-1].strip()
    return response
