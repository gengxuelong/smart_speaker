import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel, CpmTokenizer

from chatbot_pytorch import model_store
from utils_func import top_k_top_p_filtering, set_logger
from os.path import join
from flask import Flask, redirect, url_for, request

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码


@app.route('/novel', methods=['POST', 'GET'])
def novel():
    if request.method == 'POST':
        data = request.get_json()
        context = data['context']
        max_len = data['max_len']
    elif request.method == 'GET':
        context = request.args.get('context', type=str)
        max_len = request.args.get('max_len', type=int)
    print("receive request, context:{}".format(context))

    res_text = model_store.get_chat_bot_reponse(context, max_len)
    result = {"content": res_text}
    print("generated result:{}".format(result))
    return result


# --port 8085 --model_path model/zuowen_epoch40 --context_len 200
# http://localhost:8085/novel?&context='你好'&max_len=200
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8050)
