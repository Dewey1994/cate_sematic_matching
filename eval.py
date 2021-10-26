import copy

from transformers import AutoModel
from transformers import AutoConfig, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
from model import model
import pandas as pd
from config import config
import transformers
from torch.cuda.amp import GradScaler, autocast

df = pd.read_csv("data/test.csv")
st = []
for idx, i in df.iterrows():
    tmp = []
    tmp.append(i[1])
    tmp.append(i[2])
    tmp.append(i[3])
    st.append(copy.deepcopy(tmp))

if torch.cuda.is_available():
    print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    DEVICE = torch.device('cuda:0')
else:
    print("\n[INFO] GPU not found. Using CPU")
    DEVICE = torch.device('cpu')

# # config = AutoConfig("/Users/dewey/PycharmProjects/SimCSE/result/my-sup-simcse-bert-base-newcate")
tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese")

model1 = model(config, config.MODEL_PATH, 'cls').to(DEVICE)
model1.load_state_dict(torch.load('./result/model2.pth', map_location=DEVICE))
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

fuli = []
zhengli = []
q = 0
for texts in st:
    if q % 1000 == 0:
        print(q)
    q+=1
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").data
    device = DEVICE
    # Get the embeddings
    with torch.no_grad():
        # embeddings = model(**inputs).pooler_output
        ids = inputs['input_ids'].to(device, dtype=torch.long)
        mask = inputs['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.long)
        batch_size = ids.size(0)
        num_sent = ids.size(1)
        ids = ids.view((-1, ids.size(-1)))  # (bs * num_sent, len)
        mask = mask.view((-1, mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = model1(ids, mask=mask, token_type_ids=token_type_ids, batch_size=1,
                         num_sent=3)
        embeddings = outputs.squeeze().cpu()

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
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
        print("ops Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))

    print("正 Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
    print("负 Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))

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

import seaborn as sns
import matplotlib.pyplot as plt

f = plt.figure(figsize=(10, 5))

f.add_subplot(1, 1, 1)

sns.kdeplot(zhengli)

f.add_subplot(1, 2, 1)

sns.kdeplot(fuli)

plt.savefig('./result/kde.png')
