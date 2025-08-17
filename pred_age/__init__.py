"""
pred_age: 年齢予測モジュール

このモジュールはテキストデータからの年齢予測機能を提供します。
"""

# データ処理関連
from .data import (
    CreateDataset,
    ToPadding,
    created_pad,
    make_dataloader,
    make_pad,
    split_data,
    text_pad,
    to_token,
)

# 評価関連
from .metrics import evaluate, evaluation, make_thresh

# モデル関連
from .models import Algorithm, BertMultiClassificationModel, PositionalEncoding

# ユーティリティ関連
from .utils import multi_label_distribution_learning, one_hot

__version__ = "1.0.0"

__all__ = [
    # データ処理
    "ToPadding",
    "text_pad",
    "created_pad",
    "make_pad",
    "CreateDataset",
    "split_data",
    "to_token",
    "make_dataloader",
    # ラベル変換
    "multi_label_distribution_learning",
    "one_hot",
    # 評価
    "evaluation",
    "evaluate",
    "make_thresh",
    # モデル
    "PositionalEncoding",
    "BertMultiClassificationModel",
    "Algorithm",
]
