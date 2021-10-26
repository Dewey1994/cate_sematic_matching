from torch.utils.data import Dataset, DataLoader
from config import config
import torch


class BERTdataset(Dataset):
    def __init__(self, data, label, hard_neg):
        super(BERTdataset, self).__init__()
        self.data = data
        self.label = label
        self.hard_neg = hard_neg
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     data = self.data[idx]
    #     data = " ".join(data.split())
    #     label = self.label[idx]
    #     label = " ".join(label.split())
    #     hard_neg = self.hard_neg[idx]
    #     hard_neg = " ".join(hard_neg.split())
    #
    #     inputs1 = self.tokenizer.encode_plus(
    #         data,
    #         None,
    #         truncation=True,
    #         add_special_tokens=True,
    #         max_length=self.max_len,
    #         padding='max_length'
    #     )
    #
    #     ids1 = torch.tensor(inputs1['input_ids'], dtype=torch.long).view(1,-1)
    #     mask1 = torch.tensor(inputs1['attention_mask'], dtype=torch.long).view(1,-1)
    #     token_type_ids1 = torch.tensor(inputs1['token_type_ids'], dtype=torch.long).view(1,-1)
    #
    #     inputs2 = self.tokenizer.encode_plus(
    #         label,
    #         None,
    #         truncation=True,
    #         add_special_tokens=True,
    #         max_length=self.max_len,
    #         padding='max_length'
    #     )
    #
    #     ids2 = torch.tensor(inputs2['input_ids'], dtype=torch.long).view(1,-1)
    #     mask2 = torch.tensor(inputs2['attention_mask'], dtype=torch.long).view(1,-1)
    #     token_type_ids2 = torch.tensor(inputs2['token_type_ids'], dtype=torch.long).view(1,-1)
    #
    #     inputs3 = self.tokenizer.encode_plus(
    #         hard_neg,
    #         None,
    #         truncation=True,
    #         add_special_tokens=True,
    #         max_length=self.max_len,
    #         padding='max_length'
    #     )
    #
    #     ids3 = torch.tensor(inputs3['input_ids'], dtype=torch.long).view(1,-1)
    #     mask3 = torch.tensor(inputs3['attention_mask'], dtype=torch.long).view(1,-1)
    #     token_type_ids3 = torch.tensor(inputs3['token_type_ids'], dtype=torch.long).view(1,-1)
    #
    #
    #     ids = torch.cat((ids1,ids2,ids3),dim=0)
    #     mask = torch.cat((mask1,mask2,mask3),dim=0)
    #     token_type_ids = torch.cat((token_type_ids1,token_type_ids2,token_type_ids3), dim=0)
    #     return {
    #         'ids': ids,
    #         'mask': mask,
    #         "token_type_ids": token_type_ids
    #     }

    def __getitem__(self, idx):
        data = self.data[idx]
        data = " ".join(data.split())
        label = self.label[idx]
        label = " ".join(label.split())
        hard_neg = self.hard_neg[idx]
        hard_neg = " ".join(hard_neg.split())
        text = []
        text.append(data)
        text.append(label)
        text.append(hard_neg)

        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)


        return {
            'ids': ids,
            'mask': mask,
            "token_type_ids": token_type_ids
        }