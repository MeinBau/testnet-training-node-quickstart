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
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        token_buffer = []
        for line in lines:
            # 라인 단위로 토크나이징
            tokens = self.tokenizer(line, add_special_tokens=False)["input_ids"]
            token_buffer.extend(tokens)

            # 버퍼가 충분히 차면 max_seq_length 단위로 잘라서 저장
            while len(token_buffer) >= self.max_seq_length:
                chunk = token_buffer[:self.max_seq_length]
                token_buffer = token_buffer[self.max_seq_length - self.overlap_length:]

                chunk_tensor = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones_like(chunk_tensor)
                target_mask = torch.ones_like(chunk_tensor)

                samples.append({
                    "input_ids": chunk_tensor,
                    "attention_mask": attention_mask,
                    "target_mask": target_mask
                })

        # 남은 토큰 처리
        if token_buffer:
            chunk_tensor = torch.tensor(token_buffer, dtype=torch.long)
            attention_mask = torch.ones_like(chunk_tensor)
            target_mask = torch.ones_like(chunk_tensor)
            samples.append({
                "input_ids": chunk_tensor,
                "attention_mask": attention_mask,
                "target_mask": target_mask
            })

        return samples


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
