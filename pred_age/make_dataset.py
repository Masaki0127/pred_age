from typing import Any, List, Tuple, Union

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pred_age import label_tr


class CreateDataset(Dataset):
    """データセットを作成するためのクラス.

    Attributes:
        X: テキストデータ
        y: 作成日時データ
        z: パディングマスク
        w: ラベルデータ
        max_length: 最大トークン長
        tokenizer: トークナイザー
    """

    def __init__(
        self,
        X: List[List[str]],
        y: torch.Tensor,
        z: torch.Tensor,
        w: torch.Tensor,
        max_length: int,
        tokenizer: Any,
    ) -> None:
        self.X = X
        self.y = y
        self.z = z
        self.w = w
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """データセットのサイズを返す.

        Returns:
            int: データセットのサイズ
        """
        return len(self.X)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """指定されたインデックスのデータを返す.

        Args:
            index: データのインデックス

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                トークン化されたテキスト、作成日時、パディング、ラベル
        """
        text = self.X[index]
        text_token = to_token(text, self.max_length, self.tokenizer)
        created_time = self.y[index]
        padding = self.z[index]
        labels = self.w[index]
        return text_token, created_time, padding, labels


def split_data(
    text_list: List[List[str]],
    created_list: torch.Tensor,
    padding: torch.Tensor,
    label_list: List[List[int]],
    test_size: float = 0.2,
    valid_size: float = 0.2,
    valid: bool = True,
    random_state: Union[int, None] = None,
) -> Union[Tuple[List, List, List], Tuple[List, List]]:
    """データを訓練、験証、テストセットに分割する関数.

    Args:
        text_list: テキストデータのリスト
        created_list: 作成日時データ
        padding: パディングマスク
        label_list: ラベルデータのリスト
        test_size: テストセットの割合
        valid_size: 験証セットの割合
        valid: 験証セットを作成するか
        random_state: ランダムシード

    Returns:
        Union[Tuple[List, List, List], Tuple[List, List]]:
            験証セットありの場合: (訓練, 験証, テスト)
            験証セットなしの場合: (訓練, テスト)
    """
    (
        text_train_all,
        text_test,
        created_train_all,
        created_test,
        padding_train_all,
        padding_test,
        label_train_all,
        label_test,
    ) = train_test_split(
        text_list,
        created_list,
        padding,
        label_list,
        test_size=test_size,
        random_state=random_state,
    )
    if valid:
        (
            text_train,
            text_vali,
            created_train,
            created_vali,
            padding_train,
            padding_vali,
            label_train,
            label_vali,
        ) = train_test_split(
            text_train_all,
            created_train_all,
            padding_train_all,
            label_train_all,
            test_size=valid_size / (1 - test_size),
            random_state=random_state,
        )
        return (
            [text_train, created_train, padding_train, label_train],
            [text_vali, created_vali, padding_vali, label_vali],
            [text_test, created_test, padding_test, label_test],
        )

    else:
        text_train, created_train, padding_train, label_train = (
            text_train_all,
            created_train_all,
            padding_train_all,
            label_train_all,
        )
        return [text_train, created_train, padding_train, label_train], [
            text_test,
            created_test,
            padding_test,
            label_test,
        ]


def to_token(x: List[str], max_length: int, tokenizer: Any) -> torch.Tensor:
    """テキストをトークン化する関数.

    Args:
        x: トークン化対象のテキストリスト
        max_length: 最大トークン長
        tokenizer: トークナイザー

    Returns:
        torch.Tensor: トークン化されたテンソル（3次元: [1, 3, max_length]）
    """
    x = tokenizer(
        x,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    x_list = [x["input_ids"], x["token_type_ids"], x["attention_mask"]]
    return torch.stack(x_list, dim=1)


def make_dataloader(
    text: List[List[str]],
    created: torch.Tensor,
    padding: torch.Tensor,
    label: List[List[int]],
    size: int,
    batch_size: int,
    tokenizer: Any,
    max_length: int = 512,
    shuffle: bool = True,
    mldl: bool = True,
    std: float = 1,
    scaling: bool = True,
) -> DataLoader:
    """データローダーを作成する関数.

    Args:
        text: テキストデータ
        created: 作成日時データ
        padding: パディングマスク
        label: ラベルデータ
        size: ラベルのサイズ
        batch_size: バッチサイズ
        tokenizer: トークナイザー
        max_length: 最大トークン長
        shuffle: データをシャッフルするか
        mldl: MLDLを使用するか
        std: MLDLの標準偏差
        scaling: MLDLのスケーリングを実行するか

    Returns:
        DataLoader: 作成されたデータローダー
    """
    if mldl:
        label = label_tr.multi_label_distribution_learning(
            label, size, std=std, scaling=scaling
        )
    else:
        label = label_tr.one_hot(label, size)
    dataset = CreateDataset(text, created, padding, label, max_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
