from typing import List, Tuple

import torch


def text_pad(text: List[str], period: int) -> List[str]:
    """テキストリストを指定した期間にパディングまたは切り詰める関数.

    Args:
        text: パディング対象のテキストリスト
        period: 目標とする期間長

    Returns:
        パディングまたは切り詰めされたテキストリスト
    """
    if len(text) > period:
        return text[-period:]
    elif len(text) == period:
        return text
    else:
        padded_text = []
        padded_text.extend(text)
        for _ in range(period - len(text)):
            padded_text.append("")
        return padded_text


def created_pad(created: List[int], period: int) -> torch.Tensor:
    """作成日時リストを指定した期間にパディングまたは切り詰める関数.

    Args:
        created: 作成日時のリスト
        period: 目標とする期間長

    Returns:
        torch.Tensor: パディングまたは切り詰めされたテンソル（1次元: [period]）
    """
    created_tensor = torch.tensor(created, dtype=torch.long)
    if len(created) > period:
        created_tensor = created_tensor[-period:]
        first_timestamp = created_tensor[0].clone()
        created_tensor -= first_timestamp
        return created_tensor
    elif len(created) == period:
        return created_tensor
    else:
        padded_tensor = torch.cat((created_tensor, torch.zeros(period - len(created))))
        return padded_tensor.to(torch.long)


def make_pad(text: List[str], period: int) -> torch.Tensor:
    """テキストの長さに基づいてパディングマスクを作成する関数.

    Args:
        text: 対象のテキストリスト
        period: 目標とする期間長

    Returns:
        torch.Tensor: パディングマスク（1次元: [period]）
    """
    if len(text) >= period:
        return torch.ones(period)
    elif len(text) < period:
        return torch.cat((torch.ones(len(text)), torch.zeros(period - len(text))))


class ToPadding:
    """データをパディングするためのクラス.

    Attributes:
        text: テキストデータのリスト
        created: 作成日時データのリスト
        period: パディングする期間長
    """

    def __init__(
        self, text: List[List[str]], created: List[List[int]], period: int
    ) -> None:
        self.text = text
        self.created = created
        self.period = period

    def pad_data(self) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
        """全てのデータをパディングする.

        Returns:
            Tuple containing:
                - List[List[str]]: パディングされたテキストデータ
                - torch.Tensor: パディングされた作成日時データ（2次元: [batch_size, period]）
                - torch.Tensor: パディングマスク（2次元: [batch_size, period]）
        """
        padded_text = self.pad_text()
        padded_created = self.pad_created()
        padding_mask = self.make_padding()
        return padded_text, padded_created, padding_mask

    def pad_text(self) -> List[List[str]]:
        """テキストデータをパディングする.

        Returns:
            パディングされたテキストデータのリスト
        """
        return [text_pad(text_sequence, self.period) for text_sequence in self.text]

    def pad_created(self) -> torch.Tensor:
        """作成日時データをパディングする.

        Returns:
            torch.Tensor: パディングされた作成日時データ（2次元: [batch_size, period]）
        """
        return torch.stack(
            [
                created_pad(created_sequence, self.period)
                for created_sequence in self.created
            ]
        )

    def make_padding(self) -> torch.Tensor:
        """パディングマスクを作成する.

        Returns:
            torch.Tensor: パディングマスク（2次元: [batch_size, period]）
        """
        return torch.stack(
            [make_pad(text_sequence, self.period) for text_sequence in self.text]
        )
