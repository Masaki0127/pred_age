import math
from typing import Iterable, Sequence, Tuple

import numpy as np
import scipy.signal
from tqdm import tqdm


def evaluation(
    pred: Sequence[float], label: Sequence[int], height: float
) -> Tuple[float, float]:
    """Evaluate one sample by comparing peak positions to label maxima.

    Returns age error (MAE over detected peaks) and count error (MAE of number
    of peaks vs. number of label maxima).
    """
    pred = np.append(pred, 0)
    pred = np.append(0, pred)
    age_list, _ = scipy.signal.find_peaks(
        pred,
        height=height,
        threshold=None,
        distance=9,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=0.5,
        plateau_size=None,
    )
    if len(age_list) == 0:
        age_list = np.array([np.argmax(pred)])
    labelindex = np.array([i for i, x in enumerate(label) if x == max(label)])
    age_mae = 0.0
    for i in age_list:
        i -= 1
        min_age = np.min(np.abs(labelindex - i))
        age_mae += float(min_age)
    age_mae = age_mae / len(age_list)
    num_mae = float(np.abs(len(labelindex) - len(age_list)))
    return age_mae, num_mae


def evaluate(
    pred: Iterable[Sequence[float]], label: Iterable[Sequence[int]], height: float
) -> Tuple[float, float, float, float]:
    """Aggregate metrics across samples.

    Returns tuple of (age_mae, age_rmse, num_mae, num_rmse).
    """
    age_mae = 0.0
    age_rmse = 0.0
    num_mae = 0.0
    num_rmse = 0.0
    pred_list = list(pred)
    label_list = list(label)
    for i, j in zip(tqdm(pred_list), label_list):
        x, y = evaluation(i, j, height)
        age_mae += x
        age_rmse += x**2
        num_mae += y
        num_rmse += y**2
    age_mae = age_mae / len(pred_list)
    age_rmse = math.sqrt(age_rmse / len(pred_list))
    num_mae = num_mae / len(pred_list)
    num_rmse = math.sqrt(num_rmse / len(pred_list))
    return age_mae, age_rmse, num_mae, num_rmse


def make_thresh(
    pred_vali: Iterable[Sequence[float]], label_vali: Iterable[Sequence[int]]
) -> float:
    """Coarsely then finely search a threshold that minimizes count error.

    The original behavior had implicit variables; this version initializes
    them explicitly to avoid UnboundLocalError while preserving the search
    logic and early-break semantics.
    """
    min_mae = 10000.0
    ikiti = 0.0
    ikiti2 = 0.0

    pred_vali_list = list(pred_vali)
    label_vali_list = list(label_vali)

    for t in np.arange(0, 1.1, 0.1):
        num_mae = 0.0
        for i, j in zip(tqdm(pred_vali_list), label_vali_list):
            _, x = evaluation(i, j, t)
            num_mae += x
        num_mae = num_mae / len(pred_vali_list)
        if min_mae > num_mae:
            min_mae = num_mae
            ikiti = float(t)
            continue
        break
    min_mae = 10000.0
    for q in np.arange(0, 0.11, 0.01):
        num_mae = 0.0
        for i, j in zip(tqdm(pred_vali_list), label_vali_list):
            _, x = evaluation(i, j, ikiti + q)
            num_mae += x
        num_mae = num_mae / len(pred_vali_list)
        if min_mae > num_mae:
            min_mae = num_mae
            ikiti2 = float(q)
            continue
        break
    if ikiti2 == 0.0:
        for q in np.arange(0.01, 0.11, 0.01):
            num_mae = 0.0
            for i, j in zip(tqdm(pred_vali_list), label_vali_list):
                _, x = evaluation(i, j, ikiti - q)
                num_mae += x
            num_mae = num_mae / len(pred_vali_list)
            if min_mae > num_mae:
                min_mae = num_mae
                ikiti2 = -float(q)
                continue
            break
    return float(ikiti + ikiti2)
