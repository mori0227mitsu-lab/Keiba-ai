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

from course_info import COURSE_HILL, COURSE_TURN

CATEGORICAL_COLS = [
    "venue", "track_type", "condition", "sex", "jockey", "running_style",
    "day_bias", "straight_length", "trainer", "prev_pace_note",
    "turn_direction", "hill", "horse_turn_aptitude", "horse_hill_aptitude",
    "race_class", "prev_field_strength_note",
]
FEATURE_COLS = [
    # レース条件(その場で分かる/固定知識)
    "venue", "distance", "track_type", "condition", "straight_length", "day_bias",
    "turn_direction", "hill", "race_class", "class_level",
    # 馬の基本情報(その場で分かる)
    "waku", "age", "sex", "jockey", "trainer", "running_style",
    "weight_carry", "horse_weight", "weight_diff", "popularity",
    # 過去走から引っ張る情報(予測時にも分かる)
    "prev_rank", "rest_weeks", "prev_time_sec", "prev_agari_3f", "prev_corner_pos",
    "prev_pace_note", "prev_class_level", "prev_race_time_score", "prev_field_strength_note",
    # 馬ごとの右左回り・坂の得意不得意(過去の全成績から判定)
    "horse_turn_aptitude", "horse_hill_aptitude",
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
PACE_NOTE_CLOSER_UNLUCKY = "展開不利(上がり1位なのに掲示板外)"

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


def compute_field_strength_note(df: pd.DataFrame) -> pd.DataFrame:
    """「このレースで自分より先着した馬が、後の別のレースで3着以内に入っているか」
    を判定する(field_strength_note)。

    先着した相手が後に活躍していれば、今回負けていても「レベルの高い相手に
    負けていただけ」と評価できる材料になる。判定には、そのレースより後に
    行われた(chronologicalに後の)レースの結果だけを使う。
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

    for race_id, g in df.groupby("race_id"):
        g_sorted = g.sort_values("finish_rank")
        for idx, row in g_sorted.iterrows():
            ahead = g_sorted[g_sorted["finish_rank"] < row["finish_rank"]]
            if ahead.empty:
                continue
            beaten_by_future_winner = False
            for ahead_name in ahead["horse_name"]:
                hist = history_by_name.get(ahead_name)
                if hist is None:
                    continue
                future_top3 = any(
                    (chron > row["_chron_key"]) and (rank <= 3)
                    for chron, rank in hist
                )
                if future_top3:
                    beaten_by_future_winner = True
                    break
            if beaten_by_future_winner:
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
        top3 = g["finish_rank"] <= 3
        below_avg_agari = agari_rank > (n / 2)
        tags[is_leader & top3 & below_avg_agari] = PACE_NOTE_LEADER_GRIT
        tags[is_closer & (~top3) & (agari_rank == 1)] = PACE_NOTE_CLOSER_UNLUCKY
        return tags

    if {"running_style", "finish_rank"}.issubset(df.columns):
        tagged = pd.Series(PACE_NOTE_NONE, index=df.index)
        for _, g in df.groupby("race_id", group_keys=False):
            tagged.loc[g.index] = _tag_race(g)
        df["pace_note"] = tagged
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

    if "horse_name" not in df.columns:
        # 馬名が無いデータでは何もしない(既存のprev_rank等をそのまま使う)
        for col in ("prev_time_sec", "prev_agari_3f", "prev_corner_pos", "prev_class_level", "prev_race_time_score"):
            if col not in df.columns:
                df[col] = 0.0
        if "prev_pace_note" not in df.columns:
            df["prev_pace_note"] = PACE_NOTE_NONE
        if "prev_field_strength_note" not in df.columns:
            df["prev_field_strength_note"] = FIELD_NOTE_NONE
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

    if "field_strength_note" in df.columns:
        shifted_field = grouped["field_strength_note"].shift(1)
        df["prev_field_strength_note"] = shifted_field.fillna(FIELD_NOTE_NONE)

    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = fill_prev_from_history(df)
    df = compute_horse_course_aptitude(df)
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
    if "race_class" not in df.columns:
        df["race_class"] = "未勝利"
    if "turn_direction" not in df.columns:
        df["turn_direction"] = df["venue"].map(COURSE_TURN).fillna("右") if "venue" in df.columns else "右"
    if "hill" not in df.columns:
        df["hill"] = df["venue"].map(COURSE_HILL).fillna("坂なし") if "venue" in df.columns else "坂なし"
    if "horse_turn_aptitude" not in df.columns:
        df["horse_turn_aptitude"] = APTITUDE_UNKNOWN
    if "horse_hill_aptitude" not in df.columns:
        df["horse_hill_aptitude"] = APTITUDE_UNKNOWN

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


def backtest(data_path: str, n_splits: int = 5, random_state: int = 42) -> dict:
    """「そのレースを学習に使わずに予測する」形で、印ごとの的中率を検証する。

    race_id単位でグループ分割(GroupKFold)し、あるレースの結果は
    そのレースを含まないモデルで予測することで、答えを知った上で
    予測するズル(データリーク)が起きないようにしている。

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

    result = df[["race_id", "horse_num", "finish_rank", "popularity"]].copy()
    result["ai_proba"] = oof_proba

    rows = []
    for _, g in result.groupby("race_id"):
        g = g.sort_values("ai_proba", ascending=False).reset_index(drop=True)
        g["mark"] = _assign_marks_for_n(len(g))
        rows.append(g)
    marked = pd.concat(rows, ignore_index=True)

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
        "n_races": unique_races,
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


if __name__ == "__main__":
    main()
