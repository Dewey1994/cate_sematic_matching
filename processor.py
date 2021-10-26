import torch
import numpy as np

from transformers import AutoTokenizer
from model import model
from config import config

# 预处理函数
# 参数：
# x 用户jar包中，数据处理类的predictOnlineBefore函数封装的数据，类型包括str，bytes，numpy.array，kwargs
# kwargs 用户jar包中，数据处理类的predictOnlineBefore函数封装的参数
# 返回值：
# 模型执行的输入数据

if torch.cuda.is_available():
    print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    d1 = torch.device('cuda')
else:
    print("\n[INFO] GPU not found. Using CPU")
    d1 = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained("./chinese-bert-wwm-ext")


def preprocess(x, **kwargs):
    print("preprocess start")
    text = x.split('\5')
    all_ids = []
    all_mask = []
    all_token_type_ids = []

    for i in text:
        l = []
        l.append(i)
        l.append(i)
        l.append(i)
        inputs = tokenizer(l, padding='max_length', truncation=True, max_length=config.MAX_LEN,
                           return_tensors="pt").data
        ids = inputs['input_ids'].to(d1, dtype=torch.long).unsqueeze(0)
        mask = inputs['attention_mask'].to(d1, dtype=torch.long).unsqueeze(0)
        token_type_ids = inputs['token_type_ids'].to(d1, dtype=torch.long).unsqueeze(0)
        all_ids.append(ids)
        all_mask.append(mask)
        all_token_type_ids.append(token_type_ids)
    all_ids = torch.cat(all_ids, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    all_token_type_ids = torch.cat(all_token_type_ids, dim=0)
    print("preprocess end")

    return [all_ids, all_mask, all_token_type_ids]


# 后处理函数
# 参数：
# x 模型执行后的输出数据，即model(data)所得得结果
# kwargs 用户jar包中，数据处理类的predictOnlineBefore函数封装的参数
# 返回值：
# 用户jar包中，数据处理类的predictOnlineAfter函数的输入数据类型

def postprocess(x, **kwargs):
    print("postprocess start")

    embeddings = x.cpu()
    batch_size = embeddings.size(0)
    res = ""
    # cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
    # cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
    embeddings = embeddings[:, 0, :]
    for embed in embeddings:
        embed_np = embed.numpy()
        embed_sum = np.sqrt(sum(list(map(lambda x: x ** 2, embed_np))))
        embed_np /= embed_sum
        for num in embed_np:
            res += str(num)
            res += " "
        res = res.strip()
        res += '\n'
    res = res.strip()
    print("postprocess end")

    return res


# 模型加载函数，用户自定义
# 参数：
# 返回值：
# 加载好的模型，用于模型推理

def load_model():
    print('load model start')
    model = torch.load('./model2.pth', map_location=d1)
    print("load_model end")
    return model


# 自定义推理执行函数
# 参数：
# model 模型对象
# x 预处理后的数据，即preprocess函数所得得结果
# kwargs 用户jar包中，数据处理类的predictOnlineBefore函数封装的参数
# 返回值：
# 模型推理处理结果
def run_model(model, x, **kwargs):
    print("run_model start")

    # log.debug("run model")
    # seq_len = kwargs["seq_len"]
    ids = x[0]
    mask = x[1]
    token_type_ids = x[2]
    batch_size = ids.size(0)
    num_sent = ids.size(1)
    ids = ids.view((-1, ids.size(-1)))  # (bs * num_sent, len)
    mask = mask.view((-1, mask.size(-1)))  # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)
    with torch.no_grad():
        output = model(ids, mask=mask, token_type_ids=token_type_ids, batch_size=batch_size, num_sent=3)
    print("run_model end")
    return output


if __name__ == '__main__':
    m = load_model()
    x = preprocess("老师\5销售\5")
    y = run_model(m, x)
    z = postprocess(y)
    print(z)
