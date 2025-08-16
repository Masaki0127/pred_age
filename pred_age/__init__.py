"""
pred_age: 年齢予測モジュール

このモジュールはテキストデータからの年齢予測機能を提供します。
"""

# データ処理関連
from .data_make import ToPadding, created_pad, make_pad, text_pad
from .evaluate import evaluate as evaluate_func

# 評価関連
from .evaluate import evaluation, make_thresh

# ラベル変換関連
from .label_tr import multi_label_distribution_learning, one_hot

# データセット作成関連
from .make_dataset import CreateDataset, make_dataloader, split_data, to_token

# モデル関連
from .model import Algorithm, BertMultiClassificationModel, PositionalEncoding

__version__ = "1.0.0"

__all__ = [
    # データ処理
    "ToPadding",
    "text_pad",
    "created_pad",
    "make_pad",
    # ラベル変換
    "multi_label_distribution_learning",
    "one_hot",
    # 評価
    "evaluation",
    "evaluate_func",
    "make_thresh",
    # データセット
    "CreateDataset",
    "split_data",
    "to_token",
    "make_dataloader",
    # モデル
    "PositionalEncoding",
    "BertMultiClassificationModel",
    "Algorithm",
]
