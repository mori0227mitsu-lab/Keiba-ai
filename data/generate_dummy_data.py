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

RNG = np.random.default_rng(42)

JOCKEYS = [
    ("C.ルメール", 0.22), ("川田将雅", 0.20), ("武豊", 0.15),
    ("戸崎圭太", 0.14), ("横山武史", 0.13), ("福永祐一", 0.12),
    ("坂井瑠星", 0.11), ("松山弘平", 0.10), ("池添謙一", 0.09),
    ("岩田望来", 0.08),
]

TRACK_TYPES = ["芝", "ダート"]
CONDITIONS = ["良", "稍重", "重", "不良"]
VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

N_RACES = 60          # ダミーで作るレース数
HORSES_PER_RACE = 12  # 1レースあたりの出走頭数


def make_race(race_id: int) -> pd.DataFrame:
    distance = int(RNG.choice([1200, 1400, 1600, 1800, 2000, 2400]))
    track_type = RNG.choice(TRACK_TYPES, p=[0.6, 0.4])
    condition = RNG.choice(CONDITIONS, p=[0.55, 0.25, 0.15, 0.05])
    venue = RNG.choice(VENUES)

    rows = []
    # レースごとの「隠れた強さ」を割り振り、それに基づいて着順を生成する
    true_strength = RNG.normal(0, 1, HORSES_PER_RACE)

    for i in range(HORSES_PER_RACE):
        horse_num = i + 1
        waku = (horse_num - 1) // 2 + 1
        sex = RNG.choice(["牡", "牝", "セ"], p=[0.5, 0.4, 0.1])
        age = int(RNG.choice([3, 4, 5, 6, 7], p=[0.25, 0.3, 0.25, 0.15, 0.05]))
        jockey, jockey_base_win_rate = JOCKEYS[RNG.integers(0, len(JOCKEYS))]
        weight_carry = float(RNG.integers(52, 59))
        horse_weight = int(RNG.integers(420, 520))
        weight_diff = int(RNG.integers(-10, 11))
        prev_rank = int(RNG.integers(1, 15))
        rest_weeks = int(RNG.integers(1, 20))
        popularity = horse_num  # 後で強さに応じて振り直す
        odds = round(float(RNG.uniform(1.5, 50.0)), 1)

        # 隠れた強さに、騎手の実力・斤量・前走成績などを少し反映
        strength = (
            true_strength[i]
            + jockey_base_win_rate * 2.0
            - (weight_carry - 55) * 0.05
            - (prev_rank - 5) * 0.05
            + RNG.normal(0, 0.3)
        )

        rows.append({
            "race_id": race_id,
            "venue": venue,
            "distance": distance,
            "track_type": track_type,
            "condition": condition,
            "horse_num": horse_num,
            "waku": waku,
            "sex": sex,
            "age": age,
            "jockey": jockey,
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
