"""
Shared helpers for the PHM notebooks.

Only the C-MAPSS preprocessing pipeline and
the PHM scoring metrics live here, because both are too bulky to be
inline in notebook 03 and they're used in several cells. 
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# C-MAPSS column names (NASA standard layout)
INDEX_COLS = ["unit_id", "cycle"]
OP_COLS = [f"op_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_COLS + SENSOR_COLS


def load_cmapss(data_dir, dataset="FD001"):
    """Read the three FD0xx files into train_df, test_df, rul_test_series."""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(
        data_dir / f"train_{dataset}.txt", sep=r"\s+", header=None, names=ALL_COLS
    )
    test_df = pd.read_csv(
        data_dir / f"test_{dataset}.txt", sep=r"\s+", header=None, names=ALL_COLS
    )
    rul_test = pd.read_csv(data_dir / f"RUL_{dataset}.txt", sep=r"\s+", header=None)[0]
    return train_df, test_df, rul_test


def compute_train_rul(train_df, cap):
    """RUL = max_cycle_of_unit - cycle, clipped at `cap`."""
    max_cycle = train_df.groupby("unit_id")["cycle"].transform("max")
    rul = max_cycle - train_df["cycle"]
    return rul.clip(upper=cap)


def drop_constant_sensors(train_df, test_df):
    """Drop sensor columns with zero std on the train set."""
    stds = train_df[SENSOR_COLS].std()
    keep = stds[stds > 0].index.tolist()
    cols = INDEX_COLS + keep
    return keep, train_df[cols], test_df[cols]


def make_windows(df, feature_cols, rul_series, window_size):
    """Slide a fixed window over each unit's cycle sequence.

    Returns
    -------
    X : float tensor (n_windows, window_size, n_features)
    y : float tensor (n_windows,) -- RUL at the last cycle of each window
    """
    windows, targets = [], []
    for u in df["unit_id"].unique():
        order = df[df["unit_id"] == u].sort_values("cycle").index
        X = df.loc[order, feature_cols].values
        y = rul_series.loc[order].values if rul_series is not None else None
        if len(order) <= window_size:
            continue
        for s in range(0, len(order) - window_size):
            e = s + window_size
            windows.append(X[s:e])
            if y is not None:
                targets.append(y[e - 1])
    if not windows:
        return (
            torch.empty((0, window_size, len(feature_cols))),
            torch.empty((0,)),
        )
    X_out = torch.tensor(np.stack(windows), dtype=torch.float32)
    if rul_series is None:
        return X_out, torch.empty((0,))
    y_out = torch.tensor(np.array(targets), dtype=torch.float32)
    return X_out, y_out


def build_cmapss_tensors(data_dir, dataset, window_size, rul_cap):
    """End-to-end C-MAPSS pipeline.

    Returns
    -------
    X_train, y_train : tensors of windowed train data
    X_test,  y_test  : tensors of windowed test data
    n_features       : number of sensor columns kept after constant-drop
    """
    train_df, test_df, rul_test = load_cmapss(data_dir, dataset)
    train_rul = compute_train_rul(train_df, cap=rul_cap)
    kept, train_df, test_df = drop_constant_sensors(train_df, test_df)

    scaler = StandardScaler().fit(train_df[kept])
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[kept] = scaler.transform(train_df[kept])
    test_df[kept] = scaler.transform(test_df[kept])

    # Per-row test RUL: each unit's last cycle gets the supplied RUL,
    # earlier cycles count up from there, all clipped at the same cap.
    rul_test = rul_test.reset_index(drop=True)
    test_rul_rows = pd.Series(index=test_df.index, dtype=float)
    for i, u in enumerate(test_df["unit_id"].unique()):
        order = test_df[test_df["unit_id"] == u].sort_values("cycle").index
        last = float(rul_test.iloc[i])
        steps = np.arange(len(order) - 1, -1, -1, dtype=float)
        test_rul_rows.loc[order] = np.clip(last + steps, None, rul_cap)

    X_train, y_train = make_windows(train_df, kept, train_rul, window_size)
    X_test, y_test = make_windows(test_df, kept, test_rul_rows, window_size)
    return X_train, y_train, X_test, y_test, len(kept)


def cmapss_metrics(y_true, y_pred):
    """RMSE, MAE, R^2, and the asymmetric C-MAPSS score.

    The score penalises late predictions (error > 0) harder than early
    ones, matching how a maintenance team thinks about RUL: being late
    is worse than being early.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    r2 = float(r2_score(y_true, y_pred))
    penalty = np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)
    score = float(np.sum(penalty))

    return {"rmse": rmse, "mae": mae, "r2": r2, "c_mapss_score": score}
