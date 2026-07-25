# -*- coding: utf-8 -*-
"""
ダミーの競馬データを生成するスクリプト。

本物のデータ(netkeiba、JRA-VANなど)が用意できるまでの
「動作確認・練習用」のデータセットです。
実データに差し替えるときは、このスクリプトで作られる
dummy_races.csv と同じ列構成のCSVを用意すれば
そのまま train_model.py / app.py が使えます。
"""

import numpy as np
import pandas as pd

from course_info import COURSE_STRAIGHT_LENGTH, DAY_BIAS_OPTIONS, RUNNING_STYLES, VENUES

RNG = np.random.default_rng(42)

JOCKEYS = [
    ("C.ルメール", 0.22), ("川田将雅", 0.20), ("武豊", 0.15),
    ("戸崎圭太", 0.14), ("横山武史", 0.13), ("福永祐一", 0.12),
    ("坂井瑠星", 0.11), ("松山弘平", 0.10), ("池添謙一", 0.09),
    ("岩田望来", 0.08),
]

TRACK_TYPES = ["芝", "ダート"]
CONDITIONS = ["良", "稍重", "重", "不良"]
TRAINERS = [
    "美浦 手塚久", "美浦 木村", "美浦 国枝", "栗東 中内田", "栗東 友道",
    "栗東 矢作", "栗東 杉山晴", "美浦 田中博", "栗東 高野", "美浦 斎藤誠",
]

N_RACES = 60          # ダミーで作るレース数
HORSES_PER_RACE = 12  # 1レースあたりの出走頭数


def make_race(race_id: int) -> pd.DataFrame:
    distance = int(RNG.choice([1200, 1400, 1600, 1800, 2000, 2400]))
    track_type = RNG.choice(TRACK_TYPES, p=[0.6, 0.4])
    condition = RNG.choice(CONDITIONS, p=[0.55, 0.25, 0.15, 0.05])
    venue = RNG.choice(VENUES)
    straight_length = COURSE_STRAIGHT_LENGTH[venue]
    day_bias = RNG.choice(DAY_BIAS_OPTIONS)

    rows = []
    # レースごとの「隠れた強さ」を割り振り、それに基づいて着順を生成する
    true_strength = RNG.normal(0, 1, HORSES_PER_RACE)

    for i in range(HORSES_PER_RACE):
        horse_num = i + 1
        waku = (horse_num - 1) // 2 + 1
        sex = RNG.choice(["牡", "牝", "セ"], p=[0.5, 0.4, 0.1])
        age = int(RNG.choice([3, 4, 5, 6, 7], p=[0.25, 0.3, 0.25, 0.15, 0.05]))
        jockey, jockey_base_win_rate = JOCKEYS[RNG.integers(0, len(JOCKEYS))]
        running_style = RNG.choice(RUNNING_STYLES, p=[0.15, 0.35, 0.35, 0.15])
        weight_carry = float(RNG.integers(52, 59))
        horse_weight = int(RNG.integers(420, 520))
        weight_diff = int(RNG.integers(-10, 11))
        prev_rank = int(RNG.integers(1, 15))
        rest_weeks = int(RNG.integers(1, 20))
        popularity = horse_num  # 後で強さに応じて振り直す
        odds = round(float(RNG.uniform(1.5, 50.0)), 1)

        # 当日の傾向と脚質の相性(例: 先行有利の日は逃げ・先行馬が少し有利)
        style_bonus = 0.0
        if day_bias == "先行有利" and running_style in ("逃げ", "先行"):
            style_bonus = 0.4
        elif day_bias == "差し有利" and running_style in ("差し", "追い込み"):
            style_bonus = 0.4
        elif day_bias == "内有利" and waku <= 3:
            style_bonus = 0.2
        elif day_bias == "外有利" and waku >= 6:
            style_bonus = 0.2

        # 隠れた強さに、騎手の実力・斤量・前走成績・当日傾向との相性を反映
        strength = (
            true_strength[i]
            + jockey_base_win_rate * 2.0
            - (weight_carry - 55) * 0.05
            - (prev_rank - 5) * 0.05
            + style_bonus
            + RNG.normal(0, 0.3)
        )

        rows.append({
            "race_id": race_id,
            "venue": venue,
            "distance": distance,
            "track_type": track_type,
            "condition": condition,
            "straight_length": straight_length,
            "day_bias": day_bias,
            "horse_num": horse_num,
            "horse_name": f"ダミー馬{race_id:03d}_{horse_num:02d}",
            "waku": waku,
            "sex": sex,
            "age": age,
            "jockey": jockey,
            "trainer": TRAINERS[RNG.integers(0, len(TRAINERS))],
            "running_style": running_style,
            "weight_carry": weight_carry,
            "horse_weight": horse_weight,
            "weight_diff": weight_diff,
            "prev_rank": prev_rank,
            "rest_weeks": rest_weeks,
            "popularity": popularity,
            "odds": odds,
            "_strength": strength,
        })

    df = pd.DataFrame(rows)

    # 強さの順に着順(1位〜N位)を決定(レース当日の運の要素を追加)
    race_luck = RNG.normal(0, 0.6, HORSES_PER_RACE)
    df["_true_strength"] = df["_strength"] + race_luck
    df["finish_rank"] = df["_true_strength"].rank(ascending=False, method="first").astype(int)

    # 「人気」は事前の強さの推定値であり、当日の着順とは別物として作る
    # (的中しすぎる/リークするデータにならないよう、強さに大きめのノイズを乗せる)
    df["_market_strength"] = df["_strength"] + RNG.normal(0, 1.1, HORSES_PER_RACE)
    df = df.sort_values("_market_strength", ascending=False).reset_index(drop=True)
    df["popularity"] = np.arange(1, len(df) + 1)
    df["odds"] = (df["popularity"] ** 1.5 * RNG.uniform(0.8, 1.4, len(df))).round(1) + 1.0

    df = df.drop(columns=["_strength", "_true_strength", "_market_strength"])

    # 着順に応じてタイム・上がり3F・通過順を生成(結果データなので予測材料には使わない)
    base_time = distance / 1000 * 60 + RNG.normal(0, 1.0)
    df["time_sec"] = (base_time + (df["finish_rank"] - 1) * RNG.uniform(0.1, 0.4, len(df))).round(1)
    df["agari_3f"] = (34.0 + (df["finish_rank"] - 1) * 0.15 + RNG.normal(0, 0.5, len(df))).round(1)
    style_to_pos = {"逃げ": 1, "先行": 3, "差し": 7, "追い込み": 10}
    df["corner_pos"] = df["running_style"].map(style_to_pos).fillna(5).astype(int)

    df = df.sort_values("horse_num").reset_index(drop=True)
    return df


def generate(out_path: str = "dummy_races.csv", verbose: bool = True) -> pd.DataFrame:
    """ダミーデータを生成し、out_pathにCSV保存してDataFrameを返す。

    app.py から直接呼び出して「初回アクセス時に自動生成」する用途にも使う。
    """
    all_races = [make_race(rid) for rid in range(1, N_RACES + 1)]
    data = pd.concat(all_races, ignore_index=True)
    if out_path:
        data.to_csv(out_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"{len(data)}行 x {data.shape[1]}列 のダミーデータを {out_path} に保存しました")
    return data


def main():
    generate("dummy_races.csv")


if __name__ == "__main__":
    main()
