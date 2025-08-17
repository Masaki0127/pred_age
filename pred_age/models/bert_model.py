import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertModel


class PositionalEncoding(nn.Module):
    """位置エンコーディングを行うモジュール.

    Args:
        d_model: モデルの次元数
        dropout: ドロップアウト率
    """

    def __init__(self, d_model: int = 768, dropout: float = 0.2) -> None:
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

    def forward(self, x: torch.Tensor, created_list: torch.Tensor) -> torch.Tensor:
        """位置エンコーディングを適用したフォワードパス.

        Args:
            x: 入力テンソル
            created_list: 作成日時リスト

        Returns:
            torch.Tensor: 位置エンコーディングが適用されたテンソル
        """
        x = x + torch.squeeze(self.pe[created_list, :])
        return self.dropout(x)


class BertMultiClassificationModel(nn.Module):
    """マルチラベル分類用のBERTモデル.

    Args:
        numlabel: ラベル数
        model_name: 使用するモデル名
        max_length: 最大シーケンス長
        period: 期間長
    """

    def __init__(
        self, numlabel: int, model_name: str, max_length: int, period: int
    ) -> None:
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

    def forward(
        self, text: torch.Tensor, created_list: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        """モデルのフォワードパス.

        Args:
            text: 入力テキストテンソル
            created_list: 作成日時テンソル
            padding: パディングマスクテンソル

        Returns:
            torch.Tensor: モデルの出力テンソル（2次元: [batch_size, numlabel]）
        """
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
