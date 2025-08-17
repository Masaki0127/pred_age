# Fix Error TODO

## エラーの概要
```
RuntimeError: The expanded size of the tensor (8) must match the existing size (2) at non-singleton dimension 3.  Target sizes: [1, 8, 2, 8].  Tensor sizes: [1, 1, 2, 2]
```

## エラーの原因分析

### Git diffによる変更内容確認
`pred_age/models/bert_model.py` 100行目付近で以下の変更が行われている：

**変更前（正しい実装）：**
```python
attn_mask = (
    key_block.unsqueeze(1)
    .expand(B, num_heads, T, T)
    .contiguous()
    .view(B * num_heads, T, T)
)
```

**変更後（エラーの原因）：**
```python
attn_mask = (
    key_block.unsqueeze(1)
    .expand(B, num_heads, T, num_heads)  # 間違い！最後の次元がTではなくnum_heads
    .contiguous()
    .view(B * num_heads, T, num_heads)   # 間違い！最後の次元がTではなくnum_heads
)
```

### 問題の詳細
- `key_block`の形状: `(B=1, T=2, T=2)` = `(1, 2, 2)`
- `key_block.unsqueeze(1)`の形状: `(B=1, 1, T=2, T=2)` = `(1, 1, 2, 2)`
- 正しい拡張先: `(B=1, num_heads=8, T=2, T=2)` = `(1, 8, 2, 2)`
- 間違った拡張先: `(B=1, num_heads=8, T=2, num_heads=8)` = `(1, 8, 2, 8)`

最後の次元で `2` を `8` に拡張しようとしており、これは不可能（非singleton次元の拡張）。

## 修正方針

### [ ] 修正案1: attention mask作成の次元を修正
- `expand(B, num_heads, T, num_heads)` → `expand(B, num_heads, T, T)`に戻す
- `view(B * num_heads, T, num_heads)` → `view(B * num_heads, T, T)`に戻す

## 修正実施済み項目
- [x] attention mask作成の次元修正
  - `expand(B, num_heads, T, num_heads)` → `expand(B, num_heads, T, T)`に修正
  - `view(B * num_heads, T, num_heads)` → `view(B * num_heads, T, T)`に修正

## 新たに発生したエラー

### エラー2: Target size mismatch
```
ValueError: Target size (torch.Size([1, 1])) must be the same as input size (torch.Size([1, 5]))
```

**エラーの原因分析:**
- モデル出力: `[1, 5]` (batch_size=1, numlabel=5)
- ターゲット: `[1, 1]` (間違った形状)
- `pred_age/utils/label_transforms.py`での変更が影響している可能性

**Git diffでの変更内容:**
```python
# 変更前
probability_matrix = np.zeros((len(label), size))
np.array(norm.pdf(range(size), loc=label_value, scale=std))

# 変更後
probability_matrix = np.zeros((len(label), len(label)))  # 問題！
np.array(norm.pdf(range(len(label)), loc=label_value, scale=std))  # 問題！
```

### [x] 修正案2: label_transforms.pyの修正
- `multi_label_distribution_learning`関数で`len(label)`を`size`パラメータに戻す
  - `np.zeros((len(label), len(label)))` → `np.zeros((len(label), size))`
  - `range(len(label))` → `range(size)`

## 修正結果
✅ **全てのテストが成功しました！**

```
============================================ 2 passed in 17.54s ============================================
```

両方のテストケース（`test_train_w_mldl` と `test_train_w_onehot`）が正常に動作しています。
