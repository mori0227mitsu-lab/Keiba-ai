# -*- coding: utf-8 -*-
"""
競馬予想モデルの学習スクリプト。

data/dummy_races.csv (または同じ列構成の実データCSV) を読み込み、
「その馬が3着以内に入るか」を予測する分類モデルを学習して
model/model.joblib に保存します。

実データに差し替える場合は、環境変数 KEIBA_DATA_PATH や
--data 引数でCSVパスを指定してください。
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "venue", "track_type", "condition", "sex", "jockey", "running_style",
    "day_bias", "straight_length", "trainer",
]
FEATURE_COLS = [
    # レース条件(その場で分かる)
    "venue", "distance", "track_type", "condition", "straight_length", "day_bias",
    # 馬の基本情報(その場で分かる)
    "waku", "age", "sex", "jockey", "trainer", "running_style",
    "weight_carry", "horse_weight", "weight_diff", "popularity",
    # 過去走から引っ張る情報(予測時にも分かる)
    "prev_rank", "rest_weeks", "prev_time_sec", "prev_agari_3f", "prev_corner_pos",
]
TARGET_COL = "is_top3"

# 学習データには含まれるが、予測材料には使わない列
# (そのレースの結果なので、予測時点では未知)
RESULT_ONLY_COLS = ["time_sec", "agari_3f", "corner_pos", "finish_rank"]


def fill_prev_from_history(df: pd.DataFrame) -> pd.DataFrame:
    """馬名(horse_name)を手がかりに、同じ馬の「前走」情報を自動で埋める。

    race_idの順序を時系列とみなし、各馬の1つ前のレースの
    着順・タイム・上がり3F・コーナー通過順を prev_* 列に入れる。
    初出走(前走なし)の馬は0のまま(=不明を表す)。
    """
    df = df.copy()
    if "horse_name" not in df.columns:
        # 馬名が無いデータでは何もしない(既存のprev_rank等をそのまま使う)
        for col in ("prev_time_sec", "prev_agari_3f", "prev_corner_pos"):
            if col not in df.columns:
                df[col] = 0.0
        return df

    df = df.sort_values(["race_id"]).reset_index(drop=True)
    for col in ("prev_time_sec", "prev_agari_3f", "prev_corner_pos"):
        if col not in df.columns:
            df[col] = 0.0

    # 馬ごとに、1つ前のレースの結果をシフトして取り込む
    grouped = df.groupby("horse_name", sort=False)
    for src, dst in [
        ("finish_rank", "prev_rank"),
        ("time_sec", "prev_time_sec"),
        ("agari_3f", "prev_agari_3f"),
        ("corner_pos", "prev_corner_pos"),
    ]:
        if src in df.columns:
            shifted = grouped[src].shift(1)
            # 前走がある行だけ上書きする(無い行は0=不明のまま)
            df[dst] = shifted.fillna(df[dst]).fillna(0)

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = fill_prev_from_history(df)
    df[TARGET_COL] = (df["finish_rank"] <= 3).astype(int)
    return df


def build_features(df: pd.DataFrame, encoders: dict | None = None):
    df = df.copy()

    # 足りない列を既定値で補う(古い形式のデータや、予測画面からの入力に対応)
    numeric_defaults = {
        "prev_time_sec": 0.0, "prev_agari_3f": 0.0, "prev_corner_pos": 0.0,
        "prev_rank": 0, "rest_weeks": 0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
    if "trainer" not in df.columns:
        df["trainer"] = "UNK"

    fitted = encoders is None
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if fitted:
            le = LabelEncoder()
            # 未知のカテゴリに備えて "UNK" を語彙に含めておく
            values = df[col].astype(str).tolist() + ["UNK"]
            le.fit(values)
            encoders[col] = le
        else:
            le = encoders[col]

        def safe_transform(v, le=le):
            v = str(v)
            if v not in le.classes_:
                v = "UNK"
            return le.transform([v])[0]

        df[col] = df[col].apply(safe_transform)

    X = df[FEATURE_COLS]
    return X, encoders


def train(data_path: str, out_path: str, verbose: bool = True) -> dict:
    """学習を実行し、モデル一式(dict)を返す。out_pathにも保存する。

    app.py から直接呼び出して「初回アクセス時に自動学習」する用途にも使う。
    """
    df = load_data(data_path)
    X, encoders = build_features(df)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    if verbose:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"検証AUC: {auc:.3f} (0.5=ランダム, 1.0=完璧)")

    bundle = {"model": model, "encoders": encoders, "feature_cols": FEATURE_COLS}
    if out_path:
        joblib.dump(bundle, out_path)
        if verbose:
            print(f"モデルを {out_path} に保存しました")
    return bundle


def main():
    parser = argparse.ArgumentParser()
    default_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "dummy_races.csv"
    )
    parser.add_argument("--data", default=default_path, help="学習用CSVのパス")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "model.joblib"),
        help="モデルの保存先",
    )
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()
