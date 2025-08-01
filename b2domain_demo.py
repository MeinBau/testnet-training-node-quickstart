import os
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from loguru import logger
from huggingface_hub import HfApi

from dataset import TextDataset, SFTDataCollator
from utils.constants import model2template
from utils.gpu_utils import get_gpu_type

HF_USERNAME = os.environ["HF_USERNAME"]

@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int


def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=1.44e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load dataset
    dataset = TextDataset(
        file="data/financial_news.txt",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        # template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    print("Training Completed.")

# def optuna_train_wrapper(params):
#     args = LoraTrainingArguments(
#         num_train_epochs=3  ,
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=2,
#         lora_rank=params["lora_rank"],
#         lora_alpha=params["lora_alpha"],
#         lora_dropout=params["lora_dropout"],
#         learning_rate=params["learning_rate"], #optuna 문제로 추가
#     )
#     return train_lora(model_id=model_id, context_length=context_length, training_args=args)


if __name__ == "__main__":
    
    # Define training arguments for LoRA fine-tuning
    training_args = LoraTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )
    # tuner = OptunaLoraTuner(
    #     train_fn=optuna_train_wrapper,
    #     eval_fn=None,
    #     optuna_config=OptunaConfig(n_trials=10)
    # )


    # Set model ID and context length
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    context_length = 2048

    # best_params = tuner.run()
    # print("Best LoRA Hyperparameters:", best_params)

    # Start LoRA fine-tuning
    train_lora(     
        model_id=model_id, context_length=context_length, training_args=training_args
    )

    gpu_type = get_gpu_type()

    try:
        logger.info("Start to push the lora weight to the hub...")
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_name = f"{HF_USERNAME}/domain-{model_id.replace('/', '-')}"            # check whether the repo exists
        try:
             api.create_repo(
                repo_name,
                exist_ok=False, 
                repo_type="model",
             )
        except Exception:
            logger.info(
                f"Repo {repo_name} already exists. Will commit the new version."
            )

        commit_message = api.upload_folder(
            folder_path="outputs",
            repo_id=repo_name,
            repo_type="model",
        )
        # get commit hash
        commit_hash = commit_message.oid
        logger.info(f"Commit hash: {commit_hash}")
        logger.info(f"Repo name: {repo_name}")
            # submit
            # submit_task(
            #     task_id, repo_name, "qwen1.5", gpu_type, commit_hash
            # )
            
    except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
    finally:
            # cleanup merged_model and output
            os.system("rm -rf merged_model")
            os.system("rm -rf outputs")

