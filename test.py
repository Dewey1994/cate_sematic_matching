from transformers import AutoModel
from transformers import AutoConfig,AutoTokenizer
import torch
from scipy.spatial.distance import cosine
from model import model

from config import config
import transformers
from torch.cuda.amp import GradScaler, autocast


# # config = AutoConfig("/Users/dewey/PycharmProjects/SimCSE/result/my-sup-simcse-bert-base-newcate")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model1 = model(config, config.MODEL_PATH,'cls')
model1.load_state_dict(torch.load('bert-base-chinese-layer4_fold_0.pt',map_location=torch.device('cpu')))
# pass
texts = [
    "招聘中餐厨师",
    "厨师，酒店，餐饮",
    "保安，商场，零售"
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").data
device = "cpu"
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
    embeddings = outputs.squeeze()

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
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