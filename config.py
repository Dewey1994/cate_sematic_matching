import transformers

from torch.cuda.amp import GradScaler


class config:
    MODEL_NAME = 'bert-base-wwm'
    NB_EPOCHS = 20
    LR = 5e-5
    MAX_LEN = 16
    N_SPLITS = 5
    TRAIN_BS = 128
    VALID_BS = 128
    HIDDEN_SIZE = 768
    TEMP = 0.05
    HARD_NEGATIVE_WEIGHT = 1
    alpha = 4
    FILE_NAME = 'data/data_shuffle_v3.csv'
    MODEL_PATH = "./chinese-bert-wwm-ext"
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)
    scaler = GradScaler()
