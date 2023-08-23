import torch
from torch import nn
from transformers import CpmTokenizer

import utils_func
from data_handle import get_iter, get_loss, get_chat_model_tokenizer
from model_store import get_big_model


def train_seq2seq(net, tokenizer: CpmTokenizer, data_iter, lr, num_epochs, device):
    """训练序列到序列模型"""

    def xavier_init_weights(m):
        if type(m) == nn.Linear and m.requires_grad_:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU and m.requires_grad_:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = get_loss()
    animator = utils_func.Animator()
    timer = utils_func.Timer()
    logger = utils_func.get_logger()
    logger.info("start training......")
    metric = utils_func.Accumulator(2)  # 训练损失总和，词元数量
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tokenizer.bos_token_id] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学, 强制让模型输出的句子以bos开头
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进⾏“反向传播”
            utils_func.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            logger(f'epoch:{epoch} loss: {metric[0] / metric[1]:.3f}')
            animator.add_dynamic(epoch + 1, (metric[0] / metric[1]))
    logger(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} 'f'tokens/sec on {str(device)}')


if __name__ == '__main__':
    """"""
    big_model = get_big_model()
    _, tokenizer = get_chat_model_tokenizer()
    train_seq2seq(big_model, tokenizer, get_iter(), 0.001, 50, torch.device('cpu'))
