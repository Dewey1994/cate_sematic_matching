import copy

from transformers import AutoModel
from transformers import AutoConfig, AutoTokenizer
from scipy.spatial.distance import cosine

import torch
from torch.nn import CosineSimilarity
from model import model
import pandas as pd
from config import config
from BERTdataset import BERTdataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from torch.cuda.amp import GradScaler, autocast

df = pd.read_csv("./data/data_shuffle_v2.csv")

dataset = BERTdataset(
    data=df['data'].values,
    label=df['label'].values,
    hard_neg=df['hard_neg'].values
)

test = DataLoader(
    dataset,
    batch_size=config.VALID_BS,
    shuffle=False,
    num_workers=0
)
# # config = AutoConfig("/Users/dewey/PycharmProjects/SimCSE/result/my-sup-simcse-bert-base-newcate")
if torch.cuda.is_available():
    print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    DEVICE = torch.device('cuda:0')
else:
    print("\n[INFO] GPU not found. Using CPU")
    DEVICE = torch.device('cpu')
model1 = torch.load('./model.pth', map_location=DEVICE)
# pass
# texts = [
#     "招聘中餐厨师",
#     "厨师，酒店，餐饮",
#     "保安，商场，零售"
# ]
c07 = 0
c08 = 0
c09 = 0
e05 = 0
zhengli = []
fuli = []
ops = []

model1.eval()
prog_bar = tqdm(enumerate(test), total=len(test))
for idx, texts in prog_bar:

    # Get the embeddings
    with torch.no_grad():
        # embeddings = model(**inputs).pooler_output

        ids = texts['ids'].to(DEVICE, dtype=torch.long)
        mask = texts['mask'].to(DEVICE, dtype=torch.long)
        token_type_ids = texts['token_type_ids'].to(DEVICE, dtype=torch.long)

        batch_size = ids.size(0)
        num_sent = ids.size(1)
        ids = ids.view((-1, ids.size(-1)))  # (bs * num_sent, len)
        mask = mask.view((-1, mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = model1(ids, mask=mask, token_type_ids=token_type_ids, batch_size=batch_size,
                         num_sent=3)
        embeddingss = outputs.squeeze().cpu()

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    for idx, embeddings in enumerate(embeddingss):
        cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
        cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
        zhengli.append(cosine_sim_0_1)
        fuli.append(cosine_sim_0_2)
        if cosine_sim_0_1 >= 0.8:
            c08 += 1
        if cosine_sim_0_1 >= 0.9:
            c09 += 1
        if cosine_sim_0_1 >= 0.7:
            c07 += 1
        if cosine_sim_0_2 <= 0.5:
            e05 += 1
        if cosine_sim_0_2 >= 0.8:
            print("ops Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (dataset.tokenizer.decode(texts['ids'][idx][0]), dataset.tokenizer.decode(texts['ids'][idx][2]), cosine_sim_0_2))
            ops.append("ops Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (dataset.tokenizer.decode(texts['ids'][idx][0]), dataset.tokenizer.decode(texts['ids'][idx][2]), cosine_sim_0_2)
)
        # print("正 Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
    # print("负 Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))

print("阈值大于等于0.7的正例的占比为{}".format(c07 / len(df)))
print("阈值大于等于0.8的正例的占比为{}".format(c08 / len(df)))
print("阈值大于等于0.9的正例的占比为{}".format(c09 / len(df)))
print("阈值小于等于0.5的负例的占比为{}".format(e05 / len(df)))
# import torch
# from scipy.spatial.distance import cosine
# from transformers import AutoModel, AutoTokenizer
#
# # Import our models. The package will take care of downloading the models automatically
# tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
# model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
#
# # Tokenize input texts
# texts = [
#     "There's a kid on a skateboard.",
#     "A kid is skateboarding.",
#     "A kid is skateboarding."
# ]
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#
# # Get the embeddings
# with torch.no_grad():
#     embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
#
# # Calculate cosine similarities
# # Cosine similarities are in [-1, 1]. Higher means more similar
# cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
# cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
#
# print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
# print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))


with open("ops.txt", 'w', encoding='utf-8') as f:
    for i in ops:
        f.write(i)
        f.write('\n')
