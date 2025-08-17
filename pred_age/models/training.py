import os
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .bert_model import BertMultiClassificationModel


class Algorithm:
    """機械学習アルゴリズムを管理するクラス.

    Args:
        numlabel: ラベル数
        model_name: 使用するモデル名
        max_length: 最大シーケンス長
        period: 期間長
        result_path: 結果保存パス
        model_path: モデル読み込みパス
        multi_gpu: マルチGPU使用数
    """

    def __init__(
        self,
        numlabel: int,
        model_name: str = "cl-tohoku/bert-base-japanese-v2",
        max_length: int = 512,
        period: int = 15,
        result_path: Union[str, None] = None,
        model_path: Union[str, None] = None,
        multi_gpu: Union[int, bool] = False,
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # GPUがつかえたらGPUを利用
        self.model = BertMultiClassificationModel(
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

    def train(
        self,
        encoder,
        valid,
        lr: float,
        grad_accum_step: int = 1,
        early_stop_step: int = 5,
    ) -> int:
        """モデルを訓練するメソッド.

        Args:
            encoder: 訓練データローダー
            valid: 験証データローダー
            lr: 学習率
            grad_accum_step: 勾配累積ステップ数
            early_stop_step: 早期停止ステップ数

        Returns:
            int: 最適エポック数
        """
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
    def loss_exact(self, test) -> float:
        """験証データでの损失を計算するメソッド.

        Args:
            test: テストデータローダー

        Returns:
            float: 损失値
        """
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
    def predict(self, test) -> Tuple[np.ndarray, np.ndarray]:
        """テストデータで予測を実行するメソッド.

        Args:
            test: テストデータローダー

        Returns:
            Tuple[np.ndarray, np.ndarray]: 予測値と正解ラベル
        """
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
