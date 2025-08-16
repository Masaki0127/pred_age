import math
from typing import List, Tuple

import numpy as np
import scipy.signal
from tqdm import tqdm


def evaluation(
    pred: np.ndarray, label: np.ndarray, height: float
) -> Tuple[float, float]:
    """予測とラベルを比較して評価メトリクスを計算する関数.

    Args:
        pred: 予測値の配列
        label: 正解ラベルの配列
        height: ピーク検出の闾値

    Returns:
        Tuple[float, float]: 年齢のMEAと数のMEA
    """
    padded_pred = np.append(pred, 0)
    padded_pred = np.append(0, padded_pred)
    age_peaks, _ = scipy.signal.find_peaks(
        padded_pred,
        height=height,
        threshold=None,
        distance=9,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=0.5,
        plateau_size=None,
    )
    if len(age_peaks) == 0:
        age_peaks = np.array([np.argmax(padded_pred)])
    label_indices = np.array(
        [idx for idx, value in enumerate(label) if value == max(label)]
    )
    total_age_mae = 0
    for peak_idx in age_peaks:
        adjusted_peak = peak_idx - 1
        min_distance = np.min(np.abs(label_indices - adjusted_peak))
        total_age_mae += min_distance
    age_mae = total_age_mae / len(age_peaks)
    num_mae = np.abs(len(label_indices) - len(age_peaks))
    return age_mae, num_mae


def evaluate(
    pred: List[np.ndarray], label: List[np.ndarray], height: float
) -> Tuple[float, float, float, float]:
    """複数の予測とラベルを比較して平均評価メトリクスを計算する関数.

    Args:
        pred: 予測値のリスト
        label: 正解ラベルのリスト
        height: ピーク検出の闾値

    Returns:
        Tuple[float, float, float, float]: 年齢のMEA, 年齢のRMSE, 数のMEA, 数のRMSE
    """
    total_age_mae = 0
    total_age_rmse = 0
    total_num_mae = 0
    total_num_rmse = 0
    for prediction, true_label in zip(tqdm(pred), label):
        age_error, num_error = evaluation(prediction, true_label, height)
        total_age_mae += age_error
        total_age_rmse += age_error**2
        total_num_mae += num_error
        total_num_rmse += num_error**2
    age_mae = total_age_mae / len(pred)
    age_rmse = math.sqrt(total_age_rmse / len(pred))
    num_mae = total_num_mae / len(pred)
    num_rmse = math.sqrt(total_num_rmse / len(pred))
    return age_mae, age_rmse, num_mae, num_rmse


def make_thresh(pred_vali: List[np.ndarray], label_vali: List[np.ndarray]) -> float:
    """験証データを使用して最適な闾値を決定する関数.

    Args:
        pred_vali: 験証用予測値のリスト
        label_vali: 験証用ラベルのリスト

    Returns:
        float: 最適闾値
    """
    min_mae = 10000
    for threshold_candidate in np.arange(0, 1.1, 0.1):
        current_num_mae = 0
        for prediction, true_label in zip(tqdm(pred_vali), label_vali):
            _, num_error = evaluation(prediction, true_label, threshold_candidate)
            current_num_mae += num_error
        current_num_mae = current_num_mae / len(pred_vali)
        if min_mae > current_num_mae:
            min_mae = current_num_mae
            best_threshold = threshold_candidate
            continue
        break
    min_mae = 10000
    for adjustment in np.arange(0, 0.11, 0.01):
        current_num_mae = 0
        for prediction, true_label in zip(tqdm(pred_vali), label_vali):
            _, num_error = evaluation(
                prediction, true_label, best_threshold + adjustment
            )
            current_num_mae += num_error
        current_num_mae = current_num_mae / len(pred_vali)
        if min_mae > current_num_mae:
            min_mae = current_num_mae
            threshold_adjustment = adjustment
            continue
        break
    if threshold_adjustment == 0:
        for adjustment in np.arange(0.01, 0.11, 0.01):
            current_num_mae = 0
            for prediction, true_label in zip(tqdm(pred_vali), label_vali):
                _, num_error = evaluation(
                    prediction, true_label, best_threshold - adjustment
                )
                current_num_mae += num_error
            current_num_mae = current_num_mae / len(pred_vali)
            if min_mae > current_num_mae:
                min_mae = current_num_mae
                threshold_adjustment = -adjustment
                continue
            break
    return best_threshold + threshold_adjustment
