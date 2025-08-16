from typing import List

import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm


def multi_label_distribution_learning(
    label: List[List[int]], size: int, std: float = 1, scaling: bool = True
) -> torch.Tensor:
    """ラベルをマルチラベル分布学習用の確率分布に変換する関数.

    Args:
        label: ラベルデータのリスト
        size: ラベルのサイズ
        std: 標準偏差
        scaling: スケーリングを実行するか

    Returns:
        torch.Tensor: MLDL形式のラベル（2次元: [batch_size, size]）
    """
    probability_matrix = np.zeros((len(label), size))
    for batch_idx, label_list in enumerate(tqdm(label)):
        for label_value in label_list:
            probability_matrix[batch_idx] = np.maximum(
                probability_matrix[batch_idx],
                np.array(norm.pdf(range(size), loc=label_value, scale=std)),
            )
        if scaling:
            probability_matrix[batch_idx] = probability_matrix[batch_idx] / max(
                probability_matrix[batch_idx]
            )
    result_tensor = torch.from_numpy(probability_matrix).clone()
    return result_tensor


def one_hot(label: List[List[int]], size: int) -> torch.Tensor:
    """ラベルをone-hotエンコーディングする関数.

    Args:
        label: ラベルデータのリスト
        size: ラベルのサイズ

    Returns:
        torch.Tensor: one-hotエンコードされたラベル（2次元: [batch_size, size]）
    """
    onehot_matrix = torch.zeros(len(label), size)
    for batch_idx, label_indices in enumerate(tqdm(label)):
        onehot_matrix[batch_idx, label_indices] = 1
    return onehot_matrix
