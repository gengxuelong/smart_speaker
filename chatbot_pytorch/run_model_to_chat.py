import torch

from utils_func import top_k_top_p_filtering
from model_store import get_chat_model_tokenizer
import torch.nn.functional as F

def generate_next_token(input_ids, model, unk_id, context_len=100, topk=5, topp=0.99, temperature=1):
    """
    对于给定的上文，生成下一个单词
    """
    # 只根据当前位置的前context_len个token进行生成
    input_ids = input_ids[:, -context_len:]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[0, -1, :]
    next_token_logits = next_token_logits / temperature
    # 对于<unk>的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    next_token_logits[unk_id] = -float('Inf')
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
    # torch.multinomial表示从候选集合中选出无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
    next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    return next_token_id
def get_chat_bot_response(text: str, max_len):
    model, tokenizer = get_chat_model_tokenizer()
    context = text
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    device = torch.device('cpu')
    max_len = max_len
    # eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    input_ids = context_ids
    cur_len = len(input_ids)
    last_token_id = input_ids[-1]  # 已生成的内容的最后一个token
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    while True:
        next_token_id = generate_next_token(input_ids, model, unk_id)
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
        cur_len += 1
        word = tokenizer.convert_ids_to_tokens(next_token_id.item())
        # 超过最大长度，并且换行
        if cur_len >= max_len and last_token_id == 8 and next_token_id == 3:
            break
        # 超过最大长度，并且生成标点符号
        if cur_len >= max_len and word in [".", "。", "！", "!", "?", "？", ",", "，"]:
            break
        # 生成结束符
        if next_token_id == sep_id:
            break
    result = tokenizer.decode(input_ids.squeeze(0))
    content = result  # 生成的最终内容
    result = {"content": content}
    return result
