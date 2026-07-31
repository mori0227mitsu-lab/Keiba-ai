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
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from course_info import COURSE_HILL, COURSE_TURN, is_gate_sensitive_course

CATEGORICAL_COLS = [
    "venue", "track_type", "condition", "sex", "jockey", "running_style",
    "day_bias", "straight_length", "trainer", "prev_pace_note",
    "turn_direction", "hill", "horse_turn_aptitude", "horse_hill_aptitude",
    "race_class", "prev_field_strength_note", "prev_stretch_out_note",
    "horse_distance_aptitude", "gate_sensitive", "race_pace", "pace_fit",
]
FEATURE_COLS = [
    # レース条件(その場で分かる/固定知識)
    "venue", "distance", "track_type", "condition", "straight_length", "day_bias",
    "turn_direction", "hill", "race_class", "class_level", "gate_sensitive",
    "race_pace", "pace_fit",
    # 馬の基本情報(その場で分かる)
    "waku", "age", "sex", "jockey", "trainer", "running_style",
    "weight_carry", "horse_weight", "weight_diff", "popularity",
    # 過去走から引っ張る情報(予測時にも分かる)
    "prev_rank", "rest_weeks", "prev_time_sec", "prev_agari_3f", "prev_corner_pos",
    "prev_pace_note", "prev_class_level", "prev_race_time_score", "prev_field_strength_note",
    "prev_stretch_out_note",
    # 馬ごとの右左回り・坂・距離の得意不得意(過去の全成績から判定)
    "horse_turn_aptitude", "horse_hill_aptitude", "horse_distance_aptitude",
]
TARGET_COL = "is_top3"

# 学習データには含まれるが、予測材料には使わない列
# (そのレースの結果なので、予測時点では未知)
RESULT_ONLY_COLS = ["time_sec", "agari_3f", "corner_pos", "finish_rank"]

# CSVファイル自体に実際に保存されているべき「生の列」。
# prev_time_sec / prev_agari_3f / prev_corner_pos / prev_pace_note は
# fill_prev_from_history() / compute_pace_note() でその場で計算される列であり、
# CSVファイルには存在しないため、ここには含めない。
# (FEATURE_COLS全体をCSVの列チェックに使うと、常に「列不足」と誤判定され、
#  実データがダミーデータで上書きされてしまう重大なバグの原因になるので注意)
RAW_REQUIRED_COLS = [
    "venue", "distance", "track_type", "condition", "straight_length", "day_bias",
    "waku", "age", "sex", "jockey", "trainer", "running_style",
    "weight_carry", "horse_weight", "weight_diff", "popularity",
    "prev_rank", "rest_weeks",
]

PACE_NOTE_NONE = "特になし"
PACE_NOTE_LEADER_GRIT = "強い勝ち方(先行して上がり負けでも勝利)"
PACE_NOTE_LEADER_STRONG_RACE = "強い競馬(先行して上がり負けでも僅差の好走)"
PACE_NOTE_CLOSER_UNLUCKY = "展開不利(上がり1位なのに掲示板外)"
PACE_NOTE_CLOSE_MARGIN = 1.0  # 4・5着でも「僅差」とみなす、3着とのタイム差の上限(秒)

FIELD_NOTE_NONE = "特になし"
FIELD_NOTE_STRONG = "高評価(先着馬が後に好走)"


def compute_race_time_level(df: pd.DataFrame) -> pd.DataFrame:
    """レースごとに、同条件(距離・コース種別)の平均タイムと比べてどれくらい
    速かった/遅かったかを数値化する(race_time_score)。

    プラスが大きいほど「平均よりも速いタイムで決着した=レベルが高いレースだった
    可能性がある」ことを表す。マイナスは平均より遅かったことを表す。
    出走馬全体で共有する、レース単位の値になる。
    """
    df = df.copy()
    if not {"distance", "track_type", "time_sec", "race_id"}.issubset(df.columns):
        df["race_time_score"] = 0.0
        return df

    valid = df[df["time_sec"].notna() & (df["time_sec"] > 0)]
    if valid.empty:
        df["race_time_score"] = 0.0
        return df

    par_table = valid.groupby(["distance", "track_type"])["time_sec"].mean()
    race_avg_time = valid.groupby("race_id")["time_sec"].mean()

    def _score(row):
        key = (row["distance"], row["track_type"])
        rid = row["race_id"]
        if rid not in race_avg_time.index or key not in par_table.index:
            return 0.0
        return round(par_table.loc[key] - race_avg_time.loc[rid], 2)

    df["race_time_score"] = df.apply(_score, axis=1)
    return df


FIELD_STRENGTH_MIN_RIVALS = 2   # 何頭以上が後に掲示板好走していれば「高評価」にするか
FIELD_STRENGTH_TIME_GAP = 1.0   # この秒数以内の僅差だった先着馬だけをカウント対象にする
FIELD_STRENGTH_BOARD_RANK = 5   # 「掲示板」とみなす着順(1〜5着)


def compute_field_strength_note(df: pd.DataFrame) -> pd.DataFrame:
    """「このレースで自分に僅差で先着した馬が、複数頭その後の別レースで
    掲示板(5着以内)に載っているか」を判定する(field_strength_note)。

    先着1頭だけの好走では「たまたま」の可能性があるため、
    - タイム差が僅差(FIELD_STRENGTH_TIME_GAP秒以内)だった先着馬のうち、
    - FIELD_STRENGTH_MIN_RIVALS頭以上が後日、掲示板(5着以内)に載っていれば
    「高評価」と判定する。判定には、そのレースより後に行われた
    (chronologicalに後の)レースの結果だけを使う。
    """
    df = df.copy()
    required = {"horse_name", "race_id", "finish_rank"}
    if not required.issubset(df.columns):
        df["field_strength_note"] = FIELD_NOTE_NONE
        return df

    df = add_chronological_sort_key(df)
    df = df.sort_values("_chron_key").reset_index(drop=True)

    note = pd.Series(FIELD_NOTE_NONE, index=df.index)
    # 馬名ごとの (chron_key, finish_rank) の履歴をあらかじめ作っておく(高速化のため)
    history_by_name = {
        name: g[["_chron_key", "finish_rank"]].values
        for name, g in df.groupby("horse_name")
    }
    has_time = "time_sec" in df.columns

    for race_id, g in df.groupby("race_id"):
        g_sorted = g.sort_values("finish_rank")
        for idx, row in g_sorted.iterrows():
            ahead = g_sorted[g_sorted["finish_rank"] < row["finish_rank"]]
            if ahead.empty:
                continue

            my_time = row.get("time_sec") if has_time else None
            strong_count = 0
            for _, arow in ahead.iterrows():
                # タイム差が大きすぎる(僅差ではない)先着馬はカウントしない
                if has_time and pd.notna(my_time) and pd.notna(arow.get("time_sec")):
                    gap = abs(my_time - arow["time_sec"])
                    if gap > FIELD_STRENGTH_TIME_GAP:
                        continue

                hist = history_by_name.get(arow["horse_name"])
                if hist is None:
                    continue
                future_board = any(
                    (chron > row["_chron_key"]) and (rank <= FIELD_STRENGTH_BOARD_RANK)
                    for chron, rank in hist
                )
                if future_board:
                    strong_count += 1

            if strong_count >= FIELD_STRENGTH_MIN_RIVALS:
                note.at[idx] = FIELD_NOTE_STRONG

    df["field_strength_note"] = note
    return df


def add_chronological_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    """race_date(あれば)とrace_idから、時系列の並び替え用キー(_chron_key)を作る。

    - race_dateが正しく解釈できる行は、その日付を使って並べる
      (同じ日付が複数あればrace_idで細かく並べる)
    - race_dateが無い/解釈できない行(古い形式のデータ)はrace_idだけで並べる
    - 「日付が分かる行」は「日付が分からない行」より必ず後ろに来るようにする
      (これまでrace_id順で貯めてきたデータの方が、日付管理を始める前の
       古いデータである、という前提に基づく)
    """
    df = df.copy()
    if "race_date" in df.columns:
        parsed = pd.to_datetime(df["race_date"], errors="coerce")
    else:
        parsed = pd.Series(pd.NaT, index=df.index)
    has_date = parsed.notna()
    df["_chron_key"] = list(zip(has_date, parsed.fillna(pd.Timestamp.min), df["race_id"]))
    df["_parsed_date"] = parsed
    return df


APTITUDE_UNKNOWN = "データ不足"
APTITUDE_DIFF_THRESHOLD = 0.34  # 複勝率の差がこれ以上あれば「得意/不得意」と判定する目安


def compute_horse_course_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬ごとに、右回り/左回り・坂あり/坂なしの複勝率を比較し、得意不得意を判定する。

    判定には「そのレースより前の成績だけ」を使う(未来の結果を見てしまう
    データリークを防ぐため)。判定に十分な過去走(各条件1走以上)が無い場合は
    「データ不足」のままにする。
    """
    df = df.copy()
    if not {"horse_name", "venue", "finish_rank", "race_id"}.issubset(df.columns):
        df["horse_turn_aptitude"] = APTITUDE_UNKNOWN
        df["horse_hill_aptitude"] = APTITUDE_UNKNOWN
        return df

    df["turn_direction"] = df["venue"].map(COURSE_TURN).fillna("右")
    df["hill"] = df["venue"].map(COURSE_HILL).fillna("坂なし")

    df = add_chronological_sort_key(df)
    df = df.sort_values("_chron_key").reset_index(drop=True)
    is_top3 = (df["finish_rank"] <= 3).astype(int)

    turn_apt = pd.Series(APTITUDE_UNKNOWN, index=df.index)
    hill_apt = pd.Series(APTITUDE_UNKNOWN, index=df.index)

    for _, g in df.groupby("horse_name", sort=False):
        idx = g.index.tolist()
        for pos, i in enumerate(idx):
            past_idx = idx[:pos]
            if len(past_idx) < 2:
                continue  # 過去走が少なすぎる場合は判定しない

            cur_turn = df.at[i, "turn_direction"]
            cur_hill = df.at[i, "hill"]
            past_turn = df.loc[past_idx, "turn_direction"]
            past_hill = df.loc[past_idx, "hill"]
            past_top3 = is_top3.loc[past_idx]

            same_turn = past_top3[past_turn == cur_turn]
            diff_turn = past_top3[past_turn != cur_turn]
            if len(same_turn) >= 1 and len(diff_turn) >= 1:
                gap = same_turn.mean() - diff_turn.mean()
                if gap >= APTITUDE_DIFF_THRESHOLD:
                    turn_apt.at[i] = f"得意({cur_turn}回り好走歴あり)"
                elif gap <= -APTITUDE_DIFF_THRESHOLD:
                    turn_apt.at[i] = f"不得意({cur_turn}回り苦手傾向)"
                else:
                    turn_apt.at[i] = "差なし"

            same_hill = past_top3[past_hill == cur_hill]
            diff_hill = past_top3[past_hill != cur_hill]
            if len(same_hill) >= 1 and len(diff_hill) >= 1:
                gap = same_hill.mean() - diff_hill.mean()
                if gap >= APTITUDE_DIFF_THRESHOLD:
                    hill_apt.at[i] = f"得意({cur_hill}好走歴あり)"
                elif gap <= -APTITUDE_DIFF_THRESHOLD:
                    hill_apt.at[i] = f"不得意({cur_hill}苦手傾向)"
                else:
                    hill_apt.at[i] = "差なし"

    df["horse_turn_aptitude"] = turn_apt
    df["horse_hill_aptitude"] = hill_apt
    return df


RACE_PACE_SLOW = "スロー"
RACE_PACE_MIDDLE = "ミドル"
RACE_PACE_HIGH = "ハイ"


def estimate_race_pace(running_styles) -> str:
    """出走馬の脚質構成から、そのレースの想定ペースを大まかに割り出す。

    考え方(簡易版):
    - 「逃げ」馬が0頭 → 誰も引っ張らないので基本スロー
    - 「逃げ」馬が1頭だけで、「先行」も少なければ、その馬の独走になりやすくスロー
    - 「逃げ」馬が2頭以上、または「逃げ+先行」が頭数の半分以上を占める → ハイペースになりやすい
    - それ以外はミドル

    厳密な計算式ではなく、あくまで目安。実際のペースは枠順・騎手の意図・展開の綾など
    様々な要因で変わるため、参考程度に見てほしい。
    """
    styles = [s for s in running_styles if pd.notna(s)]
    n = len(styles)
    if n == 0:
        return RACE_PACE_MIDDLE

    n_nige = sum(1 for s in styles if s == "逃げ")
    n_leader = sum(1 for s in styles if s in ("逃げ", "先行"))

    if n_nige >= 2 or (n > 0 and n_leader / n >= 0.5 and n_leader >= 3):
        return RACE_PACE_HIGH
    if n_nige == 0:
        return RACE_PACE_SLOW
    if n_nige == 1 and n_leader == 1:
        # 逃げ馬が1頭だけで、競り合う先行馬が全くいない
        # → その馬の独走になりやすく、スローペースが濃厚
        return RACE_PACE_SLOW
    return RACE_PACE_MIDDLE


def pace_style_fit(race_pace: str, running_style: str) -> str:
    """想定ペースと、その馬自身の脚質の相性を判定する。

    - スローペースは、逃げ・先行が有利(前が止まらないため)
    - ハイペースは、差し・追い込みが届きやすい(前が総崩れしやすいため)
    - ミドルペース、または脚質不明の場合は「五分」とする
    """
    if pd.isna(race_pace) or pd.isna(running_style):
        return "五分"
    is_leader = running_style in ("逃げ", "先行")
    is_closer = running_style in ("差し", "追い込み")
    if race_pace == RACE_PACE_SLOW and is_leader:
        return "有利"
    if race_pace == RACE_PACE_HIGH and is_closer:
        return "有利"
    if race_pace == RACE_PACE_SLOW and is_closer:
        return "不利"
    if race_pace == RACE_PACE_HIGH and is_leader:
        return "不利"
    return "五分"


def distance_category(distance) -> str:
    """距離を「短距離/マイル/中距離/長距離」の4区分に分類する。"""
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return "不明"
    if d <= 1400:
        return "短距離"
    if d <= 1800:
        return "マイル"
    if d <= 2400:
        return "中距離"
    return "長距離"


def compute_horse_distance_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬ごとに、距離区分(短距離/マイル/中距離/長距離)ごとの複勝率を比較し、
    得意不得意を判定する(horse_distance_aptitude)。

    判定には「そのレースより前の成績だけ」を使う(データリーク防止)。
    """
    df = df.copy()
    if not {"horse_name", "distance", "finish_rank", "race_id"}.issubset(df.columns):
        df["horse_distance_aptitude"] = APTITUDE_UNKNOWN
        return df

    df["_dist_cat"] = df["distance"].apply(distance_category)

    df = add_chronological_sort_key(df)
    df = df.sort_values("_chron_key").reset_index(drop=True)
    is_top3 = (df["finish_rank"] <= 3).astype(int)

    dist_apt = pd.Series(APTITUDE_UNKNOWN, index=df.index)

    for _, g in df.groupby("horse_name", sort=False):
        idx = g.index.tolist()
        for pos, i in enumerate(idx):
            past_idx = idx[:pos]
            if len(past_idx) < 2:
                continue

            cur_cat = df.at[i, "_dist_cat"]
            past_cat = df.loc[past_idx, "_dist_cat"]
            past_top3 = is_top3.loc[past_idx]

            same_cat = past_top3[past_cat == cur_cat]
            diff_cat = past_top3[past_cat != cur_cat]
            if len(same_cat) >= 1 and len(diff_cat) >= 1:
                gap = same_cat.mean() - diff_cat.mean()
                if gap >= APTITUDE_DIFF_THRESHOLD:
                    dist_apt.at[i] = f"得意({cur_cat}好走歴あり)"
                elif gap <= -APTITUDE_DIFF_THRESHOLD:
                    dist_apt.at[i] = f"不得意({cur_cat}苦手傾向)"
                else:
                    dist_apt.at[i] = "差なし"

    df["horse_distance_aptitude"] = dist_apt
    return df


def compute_pace_note(df: pd.DataFrame) -> pd.DataFrame:
    """レースごとに上がり3Fの速さ順位を計算し、展開評価(pace_note)を付ける。

    - 先行勢(逃げ・先行)が、上がり順位は下位(切れなかった)にも関わらず3着以内
      → 「強い勝ち方」(ハイペースを我慢して押し切った可能性)
    - 差し・追い込み勢が、上がり順位1位(そのレースで一番切れた)にも関わらず4着以下
      → 「展開不利」(位置取りや詰まりで恵まれなかった可能性)
    - それ以外は「特になし」
    """
    df = df.copy()
    if "agari_3f" not in df.columns or "race_id" not in df.columns:
        df["pace_note"] = PACE_NOTE_NONE
        return df

    df["pace_note"] = PACE_NOTE_NONE

    def _tag_race(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        agari_rank = g["agari_3f"].rank(method="min", ascending=True)  # 1=最速
        tags = pd.Series(PACE_NOTE_NONE, index=g.index)
        is_leader = g["running_style"].isin(["逃げ", "先行"])
        is_closer = g["running_style"].isin(["差し", "追い込み"])
        is_win = g["finish_rank"] == 1
        is_2nd_3rd = g["finish_rank"].isin([2, 3])
        top3 = g["finish_rank"] <= 3
        below_avg_agari = agari_rank > (n / 2)

        # 4・5着でも、3着とのタイム差が僅差(PACE_NOTE_CLOSE_MARGIN秒以内)なら
        # 実質的に上位争いをしていたとみなし、「強い競馬」に含める
        # (掲示板を1つ2つ外しただけの僅差なら、次走の期待値としては
        # 3着と大差ないはず、という考え方)
        is_4th_5th = g["finish_rank"].isin([4, 5])
        third_place_time = g.loc[g["finish_rank"] == 3, "time_sec"]
        if len(third_place_time) and "time_sec" in g.columns:
            time_gap = g["time_sec"] - third_place_time.iloc[0]
            close_to_board = time_gap.notna() & (time_gap <= PACE_NOTE_CLOSE_MARGIN)
        else:
            close_to_board = pd.Series(False, index=g.index)

        # 「強い勝ち方」は1着のみ。2・3着(・僅差の4・5着)は「強い競馬」という
        # 別ラベルにする(勝ってはいないので「強い勝ち方」と呼ぶのは不正確なため)
        tags[is_leader & is_win & below_avg_agari] = PACE_NOTE_LEADER_GRIT
        tags[is_leader & is_2nd_3rd & below_avg_agari] = PACE_NOTE_LEADER_STRONG_RACE
        tags[is_leader & is_4th_5th & close_to_board & below_avg_agari] = PACE_NOTE_LEADER_STRONG_RACE
        tags[is_closer & (~top3) & (agari_rank == 1)] = PACE_NOTE_CLOSER_UNLUCKY
        return tags

    if {"running_style", "finish_rank"}.issubset(df.columns):
        tagged = pd.Series(PACE_NOTE_NONE, index=df.index)
        for _, g in df.groupby("race_id", group_keys=False):
            tagged.loc[g.index] = _tag_race(g)
        df["pace_note"] = tagged
    return df


STRETCH_OUT_NOTE_NONE = "特になし"
STRETCH_OUT_NOTE_CANDIDATE = "距離延長で変わるかも(短距離で追走できず上がり超速)"

STRETCH_OUT_MAX_DISTANCE = 1400   # これ以下の距離を「短距離」とみなす
STRETCH_OUT_AGARI_MARGIN = 0.3    # レース平均よりこの秒数以上速ければ「超速」とみなす


def compute_stretch_out_note(df: pd.DataFrame) -> pd.DataFrame:
    """短距離で後方から追走できず、それでも上がりが際立って速かった馬を判定する。

    「ただの追い込み脚質」と区別するため、以下をすべて満たす場合だけ判定する。
    - 脚質が差し・追い込み
    - 距離が短距離(STRETCH_OUT_MAX_DISTANCE以下)
    - 通過順がそのレースの後方25%
    - 上がり3Fがレース平均よりSTRETCH_OUT_AGARI_MARGIN秒以上速い
    - 3着以内に入れていない(掲示板を外している)

    あくまで「距離を伸ばせば変わるかもしれない」という仮説的な着眼点であり、
    断定的な判定ではない点に注意(単なる追い込み脚質である可能性も残る)。
    """
    df = df.copy()
    required = {"running_style", "distance", "corner_pos", "agari_3f", "finish_rank", "race_id"}
    if not required.issubset(df.columns):
        df["stretch_out_note"] = STRETCH_OUT_NOTE_NONE
        return df

    df["stretch_out_note"] = STRETCH_OUT_NOTE_NONE

    def _tag_race(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        tags = pd.Series(STRETCH_OUT_NOTE_NONE, index=g.index)
        if n < 2:
            return tags

        is_closer = g["running_style"].isin(["差し", "追い込み"])
        is_short = g["distance"] <= STRETCH_OUT_MAX_DISTANCE
        missed_board = g["finish_rank"] > 3

        # 通過順(数字が大きいほど後方)が全体の後方25%かどうか
        corner_rank = g["corner_pos"].rank(method="min", ascending=False)  # 1=最後方
        is_deep = corner_rank <= max(1, round(n * 0.25))

        # 上がり3Fがレース平均よりはっきり速いかどうか
        avg_agari = g["agari_3f"].mean()
        is_fast_agari = (avg_agari - g["agari_3f"]) >= STRETCH_OUT_AGARI_MARGIN

        cond = is_closer & is_short & missed_board & is_deep & is_fast_agari
        tags[cond] = STRETCH_OUT_NOTE_CANDIDATE
        return tags

    tagged = pd.Series(STRETCH_OUT_NOTE_NONE, index=df.index)
    for _, g in df.groupby("race_id", group_keys=False):
        tagged.loc[g.index] = _tag_race(g)
    df["stretch_out_note"] = tagged
    return df


def fill_prev_from_history(df: pd.DataFrame) -> pd.DataFrame:
    """馬名(horse_name)を手がかりに、同じ馬の「前走」情報を自動で埋める。

    race_idの順序を時系列とみなし、各馬の1つ前のレースの
    着順・タイム・上がり3F・コーナー通過順・展開評価を prev_* 列に入れる。
    初出走(前走なし)の馬は既定値のまま(=不明を表す)。
    """
    df = df.copy()
    df = compute_pace_note(df)
    df = compute_race_time_level(df)
    df = compute_field_strength_note(df)
    df = compute_stretch_out_note(df)

    if "horse_name" not in df.columns:
        # 馬名が無いデータでは何もしない(既存のprev_rank等をそのまま使う)
        for col in ("prev_time_sec", "prev_agari_3f", "prev_corner_pos", "prev_class_level", "prev_race_time_score"):
            if col not in df.columns:
                df[col] = 0.0
        if "prev_pace_note" not in df.columns:
            df["prev_pace_note"] = PACE_NOTE_NONE
        if "prev_field_strength_note" not in df.columns:
            df["prev_field_strength_note"] = FIELD_NOTE_NONE
        if "prev_stretch_out_note" not in df.columns:
            df["prev_stretch_out_note"] = STRETCH_OUT_NOTE_NONE
        return df

    df = add_chronological_sort_key(df)
    df = df.sort_values("_chron_key").reset_index(drop=True)
    for col in ("prev_time_sec", "prev_agari_3f", "prev_corner_pos", "prev_class_level", "prev_race_time_score"):
        if col not in df.columns:
            df[col] = 0.0
    if "prev_pace_note" not in df.columns:
        df["prev_pace_note"] = PACE_NOTE_NONE
    if "prev_field_strength_note" not in df.columns:
        df["prev_field_strength_note"] = FIELD_NOTE_NONE
    if "prev_stretch_out_note" not in df.columns:
        df["prev_stretch_out_note"] = STRETCH_OUT_NOTE_NONE

    # 馬ごとに、1つ前のレースの結果をシフトして取り込む
    grouped = df.groupby("horse_name", sort=False)
    for src, dst in [
        ("finish_rank", "prev_rank"),
        ("time_sec", "prev_time_sec"),
        ("agari_3f", "prev_agari_3f"),
        ("corner_pos", "prev_corner_pos"),
        ("class_level", "prev_class_level"),
        ("race_time_score", "prev_race_time_score"),
    ]:
        if src in df.columns:
            shifted = grouped[src].shift(1)
            # 前走がある行だけ上書きする(無い行は0=不明のまま)
            df[dst] = shifted.fillna(df[dst]).fillna(0)

    if "pace_note" in df.columns:
        shifted_pace = grouped["pace_note"].shift(1)
        df["prev_pace_note"] = shifted_pace.fillna(PACE_NOTE_NONE)

    # 前走からの間隔(週数)を、開催日の差から自動計算する
    # (両方の日付が分かる場合だけ計算し、片方でも不明なら0=不明のまま)
    if "_parsed_date" in df.columns:
        prev_date = grouped["_parsed_date"].shift(1)
        both_known = df["_parsed_date"].notna() & prev_date.notna()
        weeks = ((df["_parsed_date"] - prev_date).dt.days / 7).round().astype("Int64")
        df.loc[both_known, "rest_weeks"] = weeks[both_known].astype(int).clip(lower=0)

    if "field_strength_note" in df.columns:
        shifted_field = grouped["field_strength_note"].shift(1)
        df["prev_field_strength_note"] = shifted_field.fillna(FIELD_NOTE_NONE)

    if "stretch_out_note" in df.columns:
        shifted_stretch = grouped["stretch_out_note"].shift(1)
        df["prev_stretch_out_note"] = shifted_stretch.fillna(STRETCH_OUT_NOTE_NONE)

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = fill_prev_from_history(df)
    df = compute_horse_course_aptitude(df)
    df = compute_horse_distance_aptitude(df)
    df["gate_sensitive"] = df.apply(
        lambda r: "枠影響大" if is_gate_sensitive_course(r.get("venue"), r.get("distance"), r.get("track_type")) else "通常",
        axis=1,
    )
    # レースごとの脚質構成から想定ペースを判定し、各馬自身の脚質との相性も付与する
    if "race_id" in df.columns and "running_style" in df.columns:
        race_pace_map = df.groupby("race_id")["running_style"].apply(estimate_race_pace)
        df["race_pace"] = df["race_id"].map(race_pace_map)
        df["pace_fit"] = df.apply(lambda r: pace_style_fit(r.get("race_pace"), r.get("running_style")), axis=1)
    else:
        df["race_pace"] = RACE_PACE_MIDDLE
        df["pace_fit"] = "五分"
    df[TARGET_COL] = (df["finish_rank"] <= 3).astype(int)
    return df


def build_features(df: pd.DataFrame, encoders: dict | None = None):
    df = df.copy()

    # 足りない列を既定値で補う(古い形式のデータや、予測画面からの入力に対応)
    numeric_defaults = {
        "prev_time_sec": 0.0, "prev_agari_3f": 0.0, "prev_corner_pos": 0.0,
        "prev_rank": 0, "rest_weeks": 0, "class_level": 0, "prev_class_level": 0,
        "prev_race_time_score": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
    if "trainer" not in df.columns:
        df["trainer"] = "UNK"
    if "prev_pace_note" not in df.columns:
        df["prev_pace_note"] = PACE_NOTE_NONE
    if "prev_field_strength_note" not in df.columns:
        df["prev_field_strength_note"] = FIELD_NOTE_NONE
    if "prev_stretch_out_note" not in df.columns:
        df["prev_stretch_out_note"] = STRETCH_OUT_NOTE_NONE
    if "race_class" not in df.columns:
        df["race_class"] = "未勝利"
    if "turn_direction" not in df.columns:
        df["turn_direction"] = df["venue"].map(COURSE_TURN).fillna("右") if "venue" in df.columns else "右"
    if "gate_sensitive" not in df.columns:
        if {"venue", "distance", "track_type"}.issubset(df.columns):
            df["gate_sensitive"] = df.apply(
                lambda r: "枠影響大" if is_gate_sensitive_course(r.get("venue"), r.get("distance"), r.get("track_type")) else "通常",
                axis=1,
            )
        else:
            df["gate_sensitive"] = "通常"
    if "race_pace" not in df.columns or "pace_fit" not in df.columns:
        if "race_id" in df.columns and "running_style" in df.columns:
            race_pace_map = df.groupby("race_id")["running_style"].apply(estimate_race_pace)
            df["race_pace"] = df["race_id"].map(race_pace_map)
            df["pace_fit"] = df.apply(lambda r: pace_style_fit(r.get("race_pace"), r.get("running_style")), axis=1)
        else:
            df["race_pace"] = RACE_PACE_MIDDLE
            df["pace_fit"] = "五分"
    if "hill" not in df.columns:
        df["hill"] = df["venue"].map(COURSE_HILL).fillna("坂なし") if "venue" in df.columns else "坂なし"
    if "horse_turn_aptitude" not in df.columns:
        df["horse_turn_aptitude"] = APTITUDE_UNKNOWN
    if "horse_hill_aptitude" not in df.columns:
        df["horse_hill_aptitude"] = APTITUDE_UNKNOWN
    if "horse_distance_aptitude" not in df.columns:
        df["horse_distance_aptitude"] = APTITUDE_UNKNOWN

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
    # 想定外の欠損値(NaN)が残っていると学習時にエラーになるため、最後に必ず0で埋める
    # (原因不明のデータ不備があっても、アプリが完全に落ちてしまうのを防ぐための安全策)
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        import sys
        print(f"[警告] 特徴量にNaNが見つかったため0で埋めました: {nan_cols}", file=sys.stderr)
    X = X.fillna(0)

    return X, encoders


def train(data_path: str, out_path: str, verbose: bool = True) -> dict:
    """学習を実行し、モデル一式(dict)を返す。out_pathにも保存する。

    app.py から直接呼び出して「初回アクセス時に自動学習」する用途にも使う。
    """
    df = load_data(data_path)
    X, encoders = build_features(df)
    y = df[TARGET_COL]
    y_win = (df["finish_rank"] == 1).astype(int)  # 勝率(1着)予測用のターゲット

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
        print(f"検証AUC(複勝): {auc:.3f} (0.5=ランダム, 1.0=完璧)")

    # 勝率(1着)予測モデル。買い目の期待値計算(単勝・馬連・三連複など)に使う。
    win_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    win_model.fit(X, y_win)

    bundle = {
        "model": model, "win_model": win_model,
        "encoders": encoders, "feature_cols": FEATURE_COLS,
    }
    if out_path:
        joblib.dump(bundle, out_path)
        if verbose:
            print(f"モデルを {out_path} に保存しました")
    return bundle


MARKS_FOR_BACKTEST = ["◎", "○", "▲", "△", "△"]


def _assign_marks_for_n(n: int) -> list:
    if n <= 3:
        marks = MARKS_FOR_BACKTEST[:n]
    elif n <= 6:
        marks = MARKS_FOR_BACKTEST[:3]
    elif n <= 9:
        marks = MARKS_FOR_BACKTEST[:4]
    else:
        marks = MARKS_FOR_BACKTEST[:5]
    return marks + [""] * (n - len(marks))


def backtest(data_path: str, n_splits: int = 5, random_state: int = 42, age_filter: str = "全馬") -> dict:
    """「そのレースを学習に使わずに予測する」形で、印ごとの的中率を検証する。

    race_id単位でグループ分割(GroupKFold)し、あるレースの結果は
    そのレースを含まないモデルで予測することで、答えを知った上で
    予測するズル(データリーク)が起きないようにしている。

    age_filterで「2歳馬のみ」「3歳以上のみ」に絞って集計することもできる
    (他の馬齢のデータを混ぜても、対象馬齢の精度が落ちていないか確認するため)。
    学習自体は常に全データを使い、集計時の絞り込みだけ行う。

    戻り値には、印ごとの複勝率・勝率と、「1番人気を毎回買った場合」との
    比較、レース数・対象頭数などを含む。
    """
    df = load_data(data_path)
    X, encoders = build_features(df)
    y = df[TARGET_COL].values
    race_ids = df["race_id"].values

    unique_races = df["race_id"].nunique()
    n_splits = max(2, min(n_splits, unique_races))

    oof_proba = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X, y, groups=race_ids):
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_state
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_proba[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    result = df[["race_id", "horse_num", "finish_rank", "popularity", "age"]].copy()
    result["ai_proba"] = oof_proba

    rows = []
    for _, g in result.groupby("race_id"):
        g = g.sort_values("ai_proba", ascending=False).reset_index(drop=True)
        g["mark"] = _assign_marks_for_n(len(g))
        rows.append(g)
    marked = pd.concat(rows, ignore_index=True)

    # 年齢で絞り込む(学習は全データのまま、集計対象だけ絞る)
    if age_filter == "2歳馬のみ":
        marked = marked[marked["age"] == 2]
    elif age_filter == "3歳以上のみ":
        marked = marked[marked["age"] >= 3]

    def _rate(sub, cond_col, cond_val):
        top3 = (sub["finish_rank"] <= 3).mean()
        win = (sub["finish_rank"] == 1).mean()
        return {"件数": len(sub), "複勝率": round(top3, 3), "勝率": round(win, 3)}

    mark_stats = {}
    for m in ["◎", "○", "▲", "△"]:
        sub = marked[marked["mark"] == m]
        if len(sub):
            mark_stats[m] = _rate(sub, None, None)

    favorite = marked[marked["popularity"] == 1]
    favorite_stats = _rate(favorite, None, None) if len(favorite) else None

    return {
        "n_races": marked["race_id"].nunique(),
        "n_horses": len(marked),
        "mark_stats": mark_stats,
        "favorite_stats": favorite_stats,
    }


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


# --- 買い目の期待値計算(ハーヴィルの公式による近似) ---
#
# 各馬の「勝率」だけから、連対系馬券(馬連・ワイド・三連複・三連単など)の
# 的中確率を近似する古典的な方法。実際のオッズに影響される厳密な確率ではないが、
# 実務上よく使われる近似(消去法的な考え方)。

def _exacta_proba(p_i: float, p_j: float) -> float:
    """1着i・2着jになる確率(馬単)の近似値。"""
    denom = 1 - p_i
    if denom <= 0:
        return 0.0
    return p_i * (p_j / denom)


def _trifecta_proba(p_i: float, p_j: float, p_k: float) -> float:
    """1着i・2着j・3着kになる確率(三連単)の近似値。"""
    denom1 = 1 - p_i
    denom2 = 1 - p_i - p_j
    if denom1 <= 0 or denom2 <= 0:
        return 0.0
    return p_i * (p_j / denom1) * (p_k / denom2)


def quinella_proba(p_i: float, p_j: float) -> float:
    """馬連(iとjが1-2着、順不同)の的中確率。"""
    return _exacta_proba(p_i, p_j) + _exacta_proba(p_j, p_i)


def trio_proba(p_i: float, p_j: float, p_k: float) -> float:
    """三連複(i,j,kが1-3着、順不同)の的中確率。"""
    import itertools
    total = 0.0
    for a, b, c in itertools.permutations([p_i, p_j, p_k]):
        total += _trifecta_proba(a, b, c)
    return total


def wide_proba(p_i: float, p_j: float, others: list) -> float:
    """ワイド(iとjが共に3着以内、順不同)の的中確率。

    othersには、レースに出走する「i,j以外」の全馬の勝率を渡す
    (三連複の考え方を使い、iとjが3着以内に入る全パターンを足し上げる)。
    """
    total = 0.0
    for p_k in others:
        total += trio_proba(p_i, p_j, p_k)
    return total


def trifecta_proba(p_i: float, p_j: float, p_k: float) -> float:
    """三連単(1着i・2着j・3着kの着順固定)の的中確率。"""
    return _trifecta_proba(p_i, p_j, p_k)


def compute_expected_values(bets: list) -> list:
    """買い目のリスト(各要素は {"種類":str, "対象":str, "確率":float, "オッズ":float})
    から、期待値(確率×オッズ)を計算して付与する。
    """
    result = []
    for bet in bets:
        proba = bet.get("確率", 0.0)
        odds = bet.get("オッズ", 0.0)
        ev = round(proba * odds, 3) if odds else None
        result.append({**bet, "期待値": ev})
    return result


if __name__ == "__main__":
    main()
