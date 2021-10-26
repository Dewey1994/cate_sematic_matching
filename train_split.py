import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import model
from config import config
from BERTdataset import BERTdataset

import transformers
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            device
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.loss_fn = self.yield_loss
        self.device = device

    def yield_loss(self, outputs, targets):
        """
        This is the loss function for this task
        """
        loss = nn.CrossEntropyLoss()
        return loss(outputs, targets)

    def train_one_epoch(self):
        """
        This function trains the model for 1 epoch through all batches
        """
        self.model.train()
        with autocast():
            tmp_loss = 0
            for idx, inputs in enumerate(self.train_data):
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                token_type_ids = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                batch_size = ids.size(0)
                num_sent = ids.size(1)
                ids = ids.view((-1, ids.size(-1)))  # (bs * num_sent, len)
                mask = mask.view((-1, mask.size(-1)))  # (bs * num_sent len)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

                outputs = self.model(ids, mask=mask, token_type_ids=token_type_ids, batch_size=batch_size,
                                     num_sent=num_sent)

                #                 print(outputs.shape)
                #                 print(targets.shape)
                # Separate representation
                z1, z2 = outputs[:, 0], outputs[:, 1]

                # Hard negative
                if num_sent == 3:
                    z3 = outputs[:, 2]

                cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
                # Hard negative
                if num_sent >= 3:
                    z1_z3_cos = model.sim(z1.unsqueeze(1), z3.unsqueeze(0))
                    cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

                labels = torch.arange(cos_sim.size(0)).long().to(self.device)
                loss_fct = nn.CrossEntropyLoss()

                # Calculate loss with hard negatives
                if num_sent == 3:
                    # Note that weights are actually logits of weights
                    z3_weight = config.HARD_NEGATIVE_WEIGHT
                    weights = torch.tensor(
                        [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                                z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
                    ).to(self.device)
                    cos_sim = cos_sim + weights

                loss = loss_fct(cos_sim, labels)
                tmp_loss += loss.item()
                config.scaler.scale(loss).backward()
                config.scaler.step(self.optimizer)
                config.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            return tmp_loss / len(self.train_data)

    def valid_one_epoch(self):
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        self.model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in enumerate(self.valid_data):
                ids = inputs['ids'].to(self.device, dtype=torch.long)
                mask = inputs['mask'].to(self.device, dtype=torch.long)
                token_type_ids = inputs['token_type_ids'].to(self.device, dtype=torch.long)
                batch_size = ids.size(0)
                num_sent = ids.size(1)
                ids = ids.view((-1, ids.size(-1)))  # (bs * num_sent, len)
                mask = mask.view((-1, mask.size(-1)))  # (bs * num_sent len)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

                outputs = model(ids, mask=mask, token_type_ids=token_type_ids, batch_size=batch_size,
                                num_sent=num_sent)

                #                 print(outputs.shape)
                #                 print(targets.shape)
                # Separate representation
                z1, z2 = outputs[:, 0], outputs[:, 1]

                # Hard negative
                if num_sent == 3:
                    z3 = outputs[:, 2]

                cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
                # Hard negative
                if num_sent >= 3:
                    z1_z3_cos = model.sim(z1.unsqueeze(1), z3.unsqueeze(0))
                    cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

                labels = torch.arange(cos_sim.size(0)).long().to(self.device)
                loss_fct = nn.CrossEntropyLoss()

                # Calculate loss with hard negatives
                if num_sent == 3:
                    # Note that weights are actually logits of weights
                    z3_weight = config.HARD_NEGATIVE_WEIGHT
                    weights = torch.tensor(
                        [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                                z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
                    ).to(self.device)
                    cos_sim = cos_sim + weights

                loss = loss_fct(cos_sim, labels)
        print('Validation RMSE: {:.2f}'.format(loss))

        return loss

    def get_model(self):
        return self.model

    def yield_optimizer(model):
        """
        Returns optimizer for specific parameters
        """
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return transformers.AdamW(optimizer_parameters, lr=config.LR)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda')
    else:
        print("\n[INFO] GPU not found. Using CPU")
        DEVICE = torch.device('cpu')

    data = pd.read_csv(config.FILE_NAME)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data[['data', 'label', 'hard_neg']]

    # Do Kfolds training and cross validation
    kf = StratifiedKFold(n_splits=config.N_SPLITS)
    nb_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, 'idx'] = np.arange(0, len(data))
    data.loc[:, 'bins'] = pd.cut(data['idx'], bins=nb_bins, labels=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=data, y=data['bins'].values)):
        # Train for only 1 fold, you can train it for more.
        if fold != 0:
            continue
        print(f"\nFold: {fold}")
        print(f"{'-' * 20}\n")

        train_data = data.loc[train_idx]
        valid_data = data.loc[valid_idx]

        train_set = BERTdataset(
            train_data['data'].values,
            train_data['label'].values,
            train_data['hard_neg'].values,
        )

        valid_set = BERTdataset(
            valid_data['data'].values,
            valid_data['label'].values,
            valid_data['hard_neg'].values,

        )

        train = DataLoader(
            train_set,
            batch_size=config.TRAIN_BS,
            shuffle=True,
            num_workers=0
        )

        valid = DataLoader(
            valid_set,
            batch_size=config.VALID_BS,
            shuffle=False,
            num_workers=0
        )

        model = model(config, config.MODEL_PATH, "cls").to(DEVICE)
        nb_train_steps = int(len(train_data) / config.TRAIN_BS * config.NB_EPOCHS)
        optimizer = Trainer.yield_optimizer(model)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=nb_train_steps
        )

        trainer = Trainer(model, optimizer, scheduler, train, valid, DEVICE)
        t_loss = []
        v_loss = []
        best_loss = 100
        for epoch in range(1, config.NB_EPOCHS + 1):
            print(f"\n{'--' * 5} EPOCH: {epoch} {'--' * 5}\n")

            # Train for 1 epoch
            train_loss = trainer.train_one_epoch()
            t_loss.append(train_loss)
            # Validate for 1 epoch
            current_loss = trainer.valid_one_epoch()
            v_loss.append(current_loss)
            if current_loss < best_loss:
                print(f"Saving best model in this fold: {current_loss:.4f}")
                torch.save(trainer.get_model(), f"model_cp.pth")
                best_loss = current_loss

        print(f"Best CE in fold: {fold} was: {best_loss:.4f}")
        print(f"Final CE in fold: {fold} was: {current_loss:.4f}")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        ax1.set_title('fold:{}-train-loss'.format(fold))
        ax1.plot(np.arange(1, config.NB_EPOCHS + 1), t_loss)
        ax2.set_title('fold:{}-train-loss'.format(fold))
        ax2.plot(np.arange(1, config.NB_EPOCHS + 1), v_loss)
        plt.savefig(f"./fold:{fold}-loss.png")
