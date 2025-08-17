import numpy as np
import pandas as pd
from transformers import BertJapaneseTokenizer

from pred_age.data import ToPadding, make_dataloader, split_data
from pred_age.metrics.evaluation import evaluate, make_thresh
from pred_age.models import Algorithm


def build_list_cell_dataframe() -> pd.DataFrame:
    """Create a DataFrame where each cell in 'content', 'child_age', and 'created'
    stores a list containing multiple items.

    Returns:
        pd.DataFrame: DataFrame with list-valued cells.
    """
    rows = [
        {
            "content": ["絵本を読んだ", "積み木で遊んだ"],
            "child_age": [2, 2],
            "created": [1, 3],
        },
        {
            "content": ["公園で走った", "お昼寝した", "おやつを食べた"],
            "child_age": [1, 2],
            "created": [1, 2, 4],
        },
        {
            "content": ["歌を歌った", "絵を描いた"],
            "child_age": [4],
            "created": [1, 2],
        },
    ]

    return pd.DataFrame(rows)


# Example instance for convenience in tests or interactive use
example_df_with_lists: pd.DataFrame = build_list_cell_dataframe()


def test_train() -> None:
    text_list = example_df_with_lists["content"].values.tolist()
    label_list = example_df_with_lists["child_age"].values.tolist()
    created_list = example_df_with_lists["created"].values.tolist()

    flat_label = [x for row in label_list for x in row]
    min_age = min(flat_label)
    for i in range(len(label_list)):
        label_list[i] = (
            np.array(label_list[i]) - min_age
        )  # ラベルの最小値を0にする(onehotにするために)

    for i in range(len(label_list)):
        for j in range(len(label_list[i])):
            if label_list[i][j] >= 81:
                label_list[i][j] = 81

    flat_label = [x for row in label_list for x in row]
    max_age = max(flat_label) + 1

    padding = ToPadding(text_list, created_list, period=2)
    text_list, created_list, pad = padding.pad_data()
    train_set, vali_set, test_set = split_data(
        text_list, created_list, pad, label_list, random_state=1
    )
    tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v2")
    trainloader = make_dataloader(
        *train_set, size=max_age, batch_size=1, tokenizer=tokenizer, max_length=64
    )
    valiloader = make_dataloader(
        *vali_set, size=max_age, batch_size=1, tokenizer=tokenizer, max_length=64
    )
    testloader = make_dataloader(
        *test_set,
        size=max_age,
        batch_size=1,
        tokenizer=tokenizer,
        max_length=64,
        mldl=False,
    )

    result_path = "result"
    algorithm = Algorithm(
        numlabel=max_age, result_path=result_path, max_length=64, period=2
    )
    ep = algorithm.train(
        trainloader, valiloader, lr=2e-5, grad_accum_step=2, early_stop_step=0
    )
    model_path = result_path + f"/model{ep}.pth"

    algorithm = Algorithm(
        numlabel=max_age, model_path=model_path, max_length=64, period=2
    )
    pred_test, label_test = algorithm.predict(testloader)
    # Convert numpy arrays to lists for evaluation functions
    pred_test_list = [pred_test[i] for i in range(len(pred_test))]
    label_test_list = [label_test[i] for i in range(len(label_test))]
    thresh = make_thresh(pred_test_list, label_test_list)
    evaluate(pred_test_list, label_test_list, thresh)
