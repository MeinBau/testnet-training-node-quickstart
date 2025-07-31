import json
from typing import Any, Dict, List

import torch
from loguru import logger
from torch.utils.data import Dataset
from utils.tool_utils import function_formatter


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length, template):
        self.tokenizer = tokenizer
        self.system_format = template["system_format"]
        self.user_format = template["user_format"]
        self.assistant_format = template["assistant_format"]
        self.tool_format = template["tool_format"]
        self.function_format = template["function_format"]
        self.observation_format = template["observation_format"]

        self.max_seq_length = max_seq_length
        logger.info("Loading data: {}".format(file))
        with open(file, "r", encoding="utf8") as f:
            data_list = f.readlines()
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # setting system information
        if self.system_format is not None:
            system = data["system"].strip() if "system" in data.keys() else self.system

            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data["conversations"]

        input_buffer = ""
        for i in range(len(conversations)):
            role = conversations[i]["role"]
            content = conversations[i]["content"].strip()

            if role != "assistant":
                if role == "user":
                    human = self.user_format.format(
                        content=content, stop_token=self.tokenizer.eos_token
                    )
                    input_buffer += human

                elif role == "function_call":
                    tool_calls = function_formatter(json.loads(content))
                    function = self.function_format.format(content=tool_calls)
                    input_buffer += function

                elif role == "observation":
                    observation = self.observation_format.format(content=content)
                    input_buffer += observation
            else:
                assistant = self.assistant_format.format(
                    content=content, stop_token=self.tokenizer.eos_token
                )

                input_tokens = self.tokenizer.encode(
                    input_buffer, add_special_tokens=False
                )
                output_tokens = self.tokenizer.encode(
                    assistant, add_special_tokens=False
                )

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
                input_buffer = ""

        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[: self.max_seq_length]
        target_mask = target_mask[: self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        }
        return inputs


class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Find the maximum length in the batch
        lengths = [len(x["input_ids"]) for x in batch if x["input_ids"] is not None]
        # Take the maximum length in the batch, if it exceeds max_seq_length, take max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # Truncate and pad
        for x in batch:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            target_mask = x["target_mask"]
            if input_ids is None:
                logger.info("some input_ids is None")
                continue
            padding_len = batch_max_len - len(input_ids)
            # Pad
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # Truncate
            input_ids = input_ids[: self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]
            target_mask = target_mask[: self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # Convert lists to tensors to get the final model input
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
        }
        return inputs


class TextDatasetProcessor:
    """
    Plain text file을 읽어와 토큰화하고, max_seq_length에 맞춰 처리하여
    생성 모델 학습에 적합한 형태로 데이터를 가공합니다.
    SFTDataCollator와 호환되도록 "target_mask"를 포함하여 데이터를 반환합니다.
    """
    def __init__(self, tokenizer, max_seq_length: int, overlap_length: int = 0):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.overlap_length = overlap_length # 오버랩 길이 추가 (선택 사항)
        if self.overlap_length >= self.max_seq_length:
            raise ValueError("Overlap length must be less than max_seq_length.")

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        주어진 텍스트 파일을 읽어와 학습 샘플 리스트로 반환합니다.
        각 샘플은 SFTDataCollator가 기대하는 "input_ids", "attention_mask", "labels", "target_mask"를 포함합니다.
        긴 텍스트 파일 처리를 위해 줄 단위로 읽어와 max_seq_length에 맞춰 토큰화합니다.
        """
        processed_samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            current_chunk_tokens = [] # 현재 처리 중인 토큰들을 저장할 리스트

            for line in f:
                # 각 줄을 토큰화 (trim_off_token_ids를 추가하여 특정 토큰 제거 가능)
                # max_length를 명시하고 truncation=True로 설정하여 모델의 최대 길이를 초과하지 않도록 합니다.
                # 이는 이전에 발생한 (2041731 > 32768) 경고를 해결하는 핵심입니다.
                tokenized_line = self.tokenizer(
                    line.strip(), # 줄 끝 공백 제거
                    max_length=self.max_seq_length, # 토크나이저 자체에서 최대 길이 적용
                    truncation=True, # 최대 길이를 초과하면 자름
                    return_attention_mask=False,
                    add_special_tokens=True # 각 샘플에 [CLS], [SEP] 등 추가 (모델에 따라 다름)
                )["input_ids"]

                # 빈 줄이거나 토큰화 결과가 없는 경우 건너뜁니다.
                if not tokenized_line:
                    continue
                
                # 토큰화된 줄을 current_chunk_tokens에 추가
                current_chunk_tokens.extend(tokenized_line)

                # current_chunk_tokens가 max_seq_length 이상이 되면 샘플을 생성
                while len(current_chunk_tokens) >= self.max_seq_length:
                    chunk = current_chunk_tokens[:self.max_seq_length]
                    
                    # 최소 길이 설정 (기존 로직 유지)
                    if len(chunk) < 50:
                        # 짧은 청크는 버리고, 남은 토큰들을 다음 샘플에 포함
                        current_chunk_tokens = current_chunk_tokens[self.max_seq_length:]
                        continue # 다음 while 루프를 바로 실행하여 다음 청크를 검사

                    input_ids = list(chunk)
                    attention_mask = [1] * len(input_ids)

                    # Causal LM의 경우, 'labels'는 일반적으로 'input_ids'와 동일합니다.
                    # 모델 내부에서 이를 시프트하여 다음 토큰 예측에 사용합니다.
                    labels = list(chunk)

                    # 'target_mask'는 손실 계산에 포함할 토큰을 지정합니다.
                    # Causal LM에서는 일반적으로 모든 토큰에 대해 손실을 계산합니다.
                    target_mask = [1] * len(input_ids)

                    processed_samples.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "target_mask": target_mask,
                    })
                    
                    # 처리된 청크만큼 current_chunk_tokens에서 제거 (overlap_length 고려)
                    current_chunk_tokens = current_chunk_tokens[self.max_seq_length - self.overlap_length:]

            # 파일의 마지막에 남아있는 토큰들 처리 (남은 토큰이 최소 길이 이상일 경우)
            if len(current_chunk_tokens) >= 50: # 최소 길이 설정
                # 남은 토큰들을 max_seq_length에 맞게 패딩하거나 자릅니다.
                # SFTDataCollator가 패딩을 처리할 것이므로, 여기서는 단순히 포함시킵니다.
                # 필요하다면 여기에 패딩 로직을 추가할 수도 있습니다.
                final_chunk = current_chunk_tokens[:self.max_seq_length] # 너무 길면 자름 (안전장치)
                
                input_ids = list(final_chunk)
                attention_mask = [1] * len(input_ids)
                labels = list(final_chunk)
                target_mask = [1] * len(input_ids)
                
                processed_samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "target_mask": target_mask,
                })

        return processed_samples


class TextDataset(Dataset):
    """
    TextDatasetProcessor를 사용하여 전처리된 데이터를 로드하는 Dataset 클래스.
    """
    def __init__(self, file: str, tokenizer, max_seq_length: int, overlap_length: int = 0):
        self.processor = TextDatasetProcessor(tokenizer, max_seq_length, overlap_length)
        self.data = self.processor.process_file(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
