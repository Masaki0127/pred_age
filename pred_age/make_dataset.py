from typing import Sequence

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from pred_age import label_tr


class createDataset(Dataset):
    def __init__(
        self,
        X: Sequence[Sequence[str]],
        y: Sequence[torch.Tensor],
        z: Sequence[torch.Tensor],
        w: torch.Tensor,
        max_length: int,
        tokenizer,
    ):
        self.X = X
        self.y = y
        self.z = z
        self.w = w
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        text = self.X[index]
        text_token = to_token(text, self.max_length, self.tokenizer)
        created_time = self.y[index]
        padding = self.z[index]
        labels = self.w[index]
        return text_token, created_time, padding, labels


def split_data(
    text_list: Sequence[Sequence[str]],
    created_list: Sequence[torch.Tensor],
    padding: Sequence[torch.Tensor],
    label_list: Sequence[Sequence[int]],
    test_size: float = 0.2,
    valid_size: float = 0.2,
    valid: bool = True,
    random_state: int | None = None,
):
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


def to_token(x: Sequence[str], max_length: int, tokenizer) -> torch.Tensor:
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
    text: Sequence[Sequence[str]],
    created: Sequence[torch.Tensor],
    padding: Sequence[torch.Tensor],
    label: Sequence[Sequence[int]] | Sequence[int],
    size: int,
    batch_size: int,
    tokenizer,
    max_length: int = 512,
    shuffle: bool = True,
    mldl: bool = True,
    std: float = 1,
    scaling: bool = True,
):
    if mldl:
        label = label_tr.MLDL(label, size, std=std, scaling=scaling)
    else:
        label = label_tr.one_hot(label, size)
    dataset = createDataset(text, created, padding, label, max_length, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
