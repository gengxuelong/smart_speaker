import torch
from torch import nn

from chatbot_pytorch.utils_func import sequence_mask
from model_store import get_chat_model_tokenizer
from torch.utils.data import DataLoader, Dataset


def truncate_pad(line, num_steps, padding_token):
    """截断或填充⽂本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def load_data_from_disk(input_size=16, filename="data/xiaohuangji/my_formatted_movie_lines.txt"):
    """加载数据集, 得到truncate_pad的 ask_token_list,ask_len_list, reply_token_list, reply_len_list"""
    ask_list = []
    reply_list = []
    ask_len_list = []
    reply_len_list = []
    _, tokenizer = get_chat_model_tokenizer()
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            temp = line.split('|')
            if len(temp) != 2:
                continue
            ask = temp[0]
            reply = temp[1]
            ask_tokens = tokenizer.encode(ask, add_special_tokens=False)
            reply_tokens = tokenizer.encode(reply, add_special_tokens=False)
            ask_tokens = ask_tokens + [tokenizer.eos_token_id]
            reply_tokens = reply_tokens + [tokenizer.eos_token_id]
            ask_len = len(ask_tokens) if len(ask_tokens) < input_size else input_size
            reply_len = len(reply_tokens) if len(reply_tokens) < input_size else input_size
            ask_tokens = truncate_pad(ask_tokens, input_size, tokenizer.pad_token_id)
            reply_tokens = truncate_pad(reply_tokens, input_size, tokenizer.pad_token_id)
            ask_list.append(ask_tokens)
            reply_list.append(reply_tokens)
            ask_len_list.append(ask_len)
            reply_len_list.append(reply_len)
    return ask_list, ask_len_list, reply_list, reply_len_list


class ChatDataset(Dataset):
    def __init__(self, ask_list, ask_len_list, reply_list, reply_len_list):
        assert len(ask_list) == len(reply_list)
        self.ask_list = ask_list
        self.reply_list = reply_list
        self.ask_len_list = ask_len_list
        self.reply_len_list = reply_len_list

    def __getitem__(self, index):
        return list(map(lambda x: torch.tensor(x),
                        [self.ask_list[index], self.ask_len_list[index], self.reply_list[index],
                         self.reply_len_list[index]]))

    def __len__(self):
        return len(self.ask_list)


def get_iter(batch_size=16):
    ask_tokens_list, ask_len_list, reply_tokens_list, reply_len_list = load_data_from_disk()
    return DataLoader(ChatDataset(ask_tokens_list, ask_len_list, reply_tokens_list, reply_len_list),
                      batch_size=batch_size, shuffle=True)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)  # 规定需要(batch_size,vocab_size,step_nums)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def get_loss():
    return MaskedSoftmaxCELoss()
