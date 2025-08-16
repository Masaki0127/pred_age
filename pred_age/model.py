import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from transformers import BertModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=768, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(50, d_model)
        position = torch.arange(0, 50, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, created_list):
        x = x + torch.squeeze(self.pe[created_list, :])
        return self.dropout(x)


class Bertmulticlassficationmodel(nn.Module):
    def __init__(self, numlabel, model_name, max_length, period):
        super().__init__()
        self.max_length = max_length
        self.period = period
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = BertModel.from_pretrained(model_name)
        self.pos_encode = PositionalEncoding()
        encoder_layers = TransformerEncoderLayer(
            d_model=768, nhead=8, dropout=0.2, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear = nn.Linear(768, numlabel)

    def forward(self, text, created_list, padding):
        text = torch.reshape(text, (-1, 3, self.max_length))
        x = self.bert_model(
            input_ids=text[:, 0, :],
            token_type_ids=text[:, 1, :],
            attention_mask=text[:, 2, :],
        )[0][:, 0, :]  # BERTの最終層のCLSを出力
        x = torch.reshape(x, (-1, self.period, 768))
        x = self.pos_encode(x, created_list)
        x = torch.mul(x, torch.unsqueeze(padding, 2))
        # Build a 3D Bool attn_mask that blocks attention to padded key positions across heads.
        # Shape must be (batch_size * num_heads, T, T)
        B, T, _ = x.shape
        num_heads = self.transformer_encoder.layers[0].self_attn.num_heads
        key_pad = (1 - padding).bool()  # (B, T), True where padded
        key_block = key_pad.unsqueeze(1).expand(B, T, T)  # (B, T_q, T_k)
        attn_mask = (
            key_block.unsqueeze(1)
            .expand(B, num_heads, T, T)
            .contiguous()
            .view(B * num_heads, T, T)
        )
        x = self.transformer_encoder(x, mask=attn_mask)
        x = torch.bmm(torch.unsqueeze(padding, 1), x)
        x = torch.squeeze(x) / torch.unsqueeze(torch.sum(padding, 1), 1)
        output = self.linear(self.dropout1(x))
        return output


class Algorithm:
    def __init__(
        self,
        numlabel,
        model_name="cl-tohoku/bert-base-japanese-v2",
        max_length=512,
        period=15,
        result_path=None,
        model_path=None,
        multi_gpu=False,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # GPUがつかえたらGPUを利用
        self.model = Bertmulticlassficationmodel(
            numlabel, model_name, max_length, period
        )
        if model_path:
            self.model.load_state_dict(torch.load(model_path)["model_state_dict"])
        if multi_gpu > 0:
            self.model = nn.DataParallel(self.model, range(multi_gpu))
        self.multi_gpu = multi_gpu
        self.loss_list = []
        self.val_list = []
        if result_path:
            self.result_dir = result_path  # 保存場所
            # 保存場所が無かったら作成（中間ディレクトリも含めて安全に作成）
            os.makedirs(self.result_dir, exist_ok=True)

    def train(self, encoder, valid, lr, grad_accum_step=1, early_stop_step=5):
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=lr
        )  # optimizer
        criterion = nn.BCEWithLogitsLoss()
        min_score = 10000  # early_stopに使う
        early_stop = True
        ep = 1
        while early_stop:
            print(f"epoch:{ep}")
            self.model.train()  # モデルを訓練モードに
            running_loss = 0.0
            optimizer.zero_grad()
            j = 0
            for data in tqdm(encoder):
                j += 1
                text, created_list, padding, labels = data
                text = text.to(self.device)
                padding = padding.to(self.device)

                output = self.model(text, created_list, padding)
                loss = criterion(output, labels.to(self.device))
                loss = loss / grad_accum_step
                loss.backward()
                if j % grad_accum_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += loss.item()

            self.loss_list.append(running_loss / len(encoder))

            # 検証 & early_stop
            score = self.loss_exact(valid)
            self.val_list.append(score)
            if min_score > score:
                min_score = score
                s = 0
            else:
                s += 1
            if s == early_stop_step:
                early_stop = False
            save_path = self.result_dir + f"/model{ep}.pth"
            if self.multi_gpu > 0:
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_path,
                )
            else:
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_path,
                )
            print(running_loss / len(encoder))
            ep += 1
        print(f"train_loss:{self.loss_list}")
        print(f"validation_loss:{self.val_list}")
        print("Finished Training")
        print(f"Saved checkpoint at {save_path}")
        return ep - (early_stop_step + 1)

    @torch.no_grad()
    def loss_exact(self, test):
        self.model.eval()
        all = 0
        vali_loss = 0
        criterion = nn.BCEWithLogitsLoss()
        for data in tqdm(test):
            text, created_list, padding, labels = data
            text = text.to(self.device)
            padding = padding.to(self.device)
            output = self.model(text, created_list, padding)
            loss = criterion(output, labels.to(self.device))
            all += len(data)
            vali_loss += loss.item()
        print(f"vali_loss:{vali_loss / len(test)}")
        return vali_loss / len(test)

    @torch.no_grad()
    def predict(self, test):
        self.model.to(self.device)
        self.model.eval()
        pred_list = None
        label_list = None
        for data in tqdm(test):
            text, created_list, padding, labels = data
            text = text.to(self.device)
            padding = padding.to(self.device)
            output = self.model(text, created_list, padding)
            outputs = torch.sigmoid(output)
            outputs = outputs.to("cpu").detach().numpy()
            labels = labels.to("cpu").detach().numpy()
            labels = np.array(labels)
            if pred_list is not None:
                pred_list = np.append(pred_list, outputs, axis=0)
                label_list = np.append(label_list, labels, axis=0)
            else:
                pred_list = np.array(outputs)
                label_list = np.array(labels)
        return pred_list, label_list
