import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from model import model
from config import  config


if torch.cuda.is_available():
    print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    DEVICE = torch.device('cuda:0')
else:
    print("\n[INFO] GPU not found. Using CPU")
    DEVICE = torch.device('cpu')

# # config = AutoConfig("/Users/dewey/PycharmProjects/SimCSE/result/my-sup-simcse-bert-base-newcate")
tokenizer = AutoTokenizer.from_pretrained("./chinese-bert-wwm-ext")

# model1 = model(config, config.MODEL_PATH, 'cls').to(DEVICE)
# model1.load_state_dict(torch.load('./model2.pth', map_location=DEVICE))
model1 = torch.load('./model.pth', map_location=DEVICE)
# Tokenize input texts
texts = [
    "体育老师/教练",
    "健身教练",
    "后厨"
]
# Get the embeddings
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").data

    # Get the embeddings
with torch.no_grad():
    # embeddings = model(**inputs).pooler_output
    ids = inputs['input_ids'].to(DEVICE, dtype=torch.long)
    mask = inputs['attention_mask'].to(DEVICE, dtype=torch.long)
    token_type_ids = inputs['token_type_ids'].to(DEVICE, dtype=torch.long)
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