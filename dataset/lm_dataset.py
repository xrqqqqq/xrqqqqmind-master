from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    # __getitem__
    # 拿到jsonl里的每一行
    def __getitem__(self, index):
        sample = self.samples[index]
        # tokenizer把文本转化成input_id
        tokens = self.tokenizer(
            str(
                sample["text"]
            ),  # jsonl里的一个text字段，转化成字符串，传入tokenizer进行编码
            add_special_tokens=False,  # 添加特殊标记
            max_length=self.max_length - 2,  # 预留BOS和EOS的位置
            truncation=True,  # 如果文本超过max_length，进行截断
        ).input_ids
        # 需要加上EOS，BOS，以及PAD的填充
        input_ids = (
            [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        )
        # 计算需要填充的PAD数量
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # PAD->编码为-100,crossloss会忽略-100的标签位置，不参与loss计算
        # 需要编写labels,防止PAD加入loss计算，错位自回归会自动把input_ids错位一格作为labels
        labels = input_ids.clone()
        labels[
            labels == self.tokenizer.pad_token_id
        ] = -100  # 把PAD token的标签置为-100，告诉模型这些位置不参与 loss 计算
        # 需要编写attention_mask。告诉模型哪些位置是有效的，哪些位置是PAD
        attention_mask = (
            input_ids != self.tokenizer.pad_token_id
        ).long()  # 有效位置为1，PAD位置为0
        # 输出的是input_ids,attention_mask.labels
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
