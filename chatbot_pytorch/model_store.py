import math

import torch
from torch import nn
from utils_func import masked_softmax, top_k_top_p_filtering
from transformers import AutoModel, GPT2LMHeadModel, AutoTokenizer, CpmTokenizer
import torch.nn.functional as F


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接⼝"""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接⼝"""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    """⽤于序列到序列学习的循环神经⽹络编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌⼊层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经⽹络模型中，第⼀个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(Decoder):
    """⽤于序列到序列学习的循环神经⽹络解码器"""

    def __init__(self, vocab_size=30000, embed_size=128, num_hiddens=128, num_layers=2,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # ⼴播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)  # 使用encode的最后一个step的状态作为上下文,重复X_steps次分别拼接至X
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)  # output中每一步都对应一个预测值,组成一个输出序列,每个预测值参考的状态都是在其前面的
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class AttentionDecoder(Decoder):
    """带有注意⼒机制解码器的基本接⼝"""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class AdditiveAttention(nn.Module):
    """加性注意⼒
    也就是: 对于value的权重系数的计算方式是:
    Q和K->线性变换至hidden维度,再相加! 最后再线性变换至1个单个值
    """

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        :param queries: (batch_size, query_nums, query_size)
        :param keys:   (batch_size, kv_nums, key_size)
        :param values:  (batch_size, kv_nums, value_size)  key_nums = value_nums = kv_num
        :param valid_lens: (batch_size,) 指明每个句子的有效长度是多少
        :return:
        """
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使⽤⼴播⽅式进⾏求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数) valid_lens:(batch_size,)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)  # 批量矩阵乘法


class DotProductAttention(nn.Module):
    """缩放点积注意⼒
    参数权重由K和Q点积得到,这就要求K和Qsize相同
    """

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class Seq2SeqAttentionDecoder(AttentionDecoder):
    """带有注意力机制的解码器"""

    def __init__(self, vocab_size=30000, embed_size=128, num_hiddens=128, num_layers=2,
                 dropout=0.08976, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """outputs的形状为(batch_size，num_steps，num_hiddens).
        hidden_state的形状为(num_layers，batch_size，num_hiddens)"""
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]


class PreTrainSeq2seqEncoder(Encoder):
    def __init__(self):
        super(PreTrainSeq2seqEncoder, self).__init__()
        model, _ = get_chat_model_tokenizer()
        self.transformer = model.transformer
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.dense = nn.Linear(768, 128)

    def forward(self, X, *args):  # X:(batch,seq)
        X = self.transformer(X)  # (batch,seq,768)
        X = X.last_hidden_state
        X = self.dense(X)  # (batch,seq,128)
        state = X[:, -1, :].unsqueeze(1)  # (batch,1,128)
        state = torch.cat([state, state], dim=1)
        return X.permute(1, 0, 2), state.permute(1, 0, 2)


model = GPT2LMHeadModel.from_pretrained("model/novel/epoch50")
tokenizer = CpmTokenizer('vocab/chinese_vocab.model')


def get_chat_model_tokenizer():
    """得到预训练的model和tokenizer"""
    return model, tokenizer


def get_big_model():
    # data = torch.ones(100, 20, dtype=torch.int64)
    encoder = PreTrainSeq2seqEncoder()
    decoder = Seq2SeqAttentionDecoder()
    model = EncoderDecoder(encoder, decoder)
    # inputs = torch.ones(100, 20, dtype=torch.int64)
    # valid_len_x = torch.ones(100, dtype=torch.int64)
    # print(model(inputs, inputs, valid_len_x)[0].shape)
    return model
