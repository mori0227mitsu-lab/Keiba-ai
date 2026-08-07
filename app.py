# -*- coding: utf-8 -*-
"""
競馬予想AI - Streamlit Webアプリ

使い方:
    streamlit run app.py

出走馬の情報を表として入力すると、学習済みモデルが
各馬の「複勝(3着以内)確率」を予測してランキング表示します。
"""

import hashlib
import io
import os
import re

import joblib
import pandas as pd
import streamlit as st

from model.train_model import (
    FEATURE_COLS, RAW_REQUIRED_COLS, build_features, compute_pace_note,
    PACE_NOTE_NONE, APTITUDE_UNKNOWN, backtest, add_chronological_sort_key,
    FIELD_NOTE_NONE, compute_race_time_level, compute_field_strength_note,
    STRETCH_OUT_NOTE_NONE, compute_stretch_out_note,
    quinella_proba, wide_proba, trio_proba, trifecta_proba, compute_expected_values,
    distance_category, compute_horse_distance_aptitude,
    PACE_NOTE_LEADER_GRIT, PACE_NOTE_LEADER_STRONG_RACE,
    estimate_race_pace, pace_style_fit, RACE_PACE_MIDDLE,
    derive_probas,
)
from data_collector import (
    parse_netkeiba_result, parse_netkeiba_results_multi, apply_corner_section_to_df,
    fetch_netkeiba_text, fetch_and_parse_netkeiba_result, fetch_and_parse_netkeiba_shutuba,
)
from github_sync import append_rows_to_csv, fetch_csv, find_existing_race_ids, get_next_race_id, update_csv
from race_class import RACE_CLASS_PATTERNS, describe_race_level
from course_info import (
    COURSE_DISTANCES,
    COURSE_HILL,
    COURSE_STRAIGHT_LENGTH,
    COURSE_TURN,
    DAY_BIAS_OPTIONS,
    RUNNING_STYLES,
    VENUES,
    is_gate_sensitive_course,
)

FEATURE_HASH = hashlib.md5(",".join(FEATURE_COLS).encode()).hexdigest()[:8]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.joblib")

DEFAULT_ROWS = pd.DataFrame([
    {"horse_num": 1, "horse_name": "サンプルホースA", "waku": 1, "sex": "牡", "age": 4,
     "jockey": "C.ルメール", "trainer": "美浦 手塚久", "running_style": "先行",
     "weight_carry": 57.0, "horse_weight": 480, "weight_diff": 2,
     "prev_rank": 2, "rest_weeks": 5, "popularity": 1, "odds": 2.5},
    {"horse_num": 2, "horse_name": "サンプルホースB", "waku": 1, "sex": "牝", "age": 3,
     "jockey": "川田将雅", "trainer": "栗東 中内田", "running_style": "差し",
     "weight_carry": 54.0, "horse_weight": 452, "weight_diff": -4,
     "prev_rank": 5, "rest_weeks": 8, "popularity": 2, "odds": 5.1},
    {"horse_num": 3, "horse_name": "サンプルホースC", "waku": 2, "sex": "牡", "age": 5,
     "jockey": "武豊", "trainer": "栗東 友道", "running_style": "逃げ",
     "weight_carry": 58.0, "horse_weight": 498, "weight_diff": 0,
     "prev_rank": 1, "rest_weeks": 4, "popularity": 3, "odds": 6.8},
])


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dummy_races.csv")

HORSE_TABLE_COLS = [
    "horse_num", "horse_name", "waku", "sex", "age", "jockey", "trainer",
    "running_style", "weight_carry", "horse_weight", "weight_diff",
    "prev_rank", "rest_weeks", "popularity", "odds",
]


NETKEIBA_SHUTUBA_PATTERN = re.compile(
    r"(?P<waku>\d{1,2})\t(?P<horse_num>\d{1,2})\t\s*\n"
    r"(?:--|✓|☆|★|◎|○|▲|△)?\s*\n"
    r"(?P<name>[^\n]+)\n"
    r"(?P<sex>[牡牝セ])(?P<age>\d{1,2})\t(?P<weight_carry>[\d.]+)\t"
    r"(?P<jockey>[^\t\n]+)\t(?P<stable>[^\t\n]+)\t"
    r"(?:(?P<horse_weight>\d{2,3})\((?P<weight_diff>[+-]?\d+)\)|計不|--)?\t"
    r"(?P<odds>[\d.]*)\t(?P<popularity>\d{1,2})?\t?"
)


def parse_netkeiba_shutuba(text: str) -> pd.DataFrame:
    """netkeibaの出馬表ページをそのままコピペした生テキストを解析する。

    (騎手名の先頭についた▲△★☆などの斤量減量マークはそのまま残す)
    ページのレイアウトが変わると解析できなくなる可能性があるので、
    その場合はエラーメッセージを出してCSV貼り付けにフォールバックしてもらう。
    """
    matches = list(NETKEIBA_SHUTUBA_PATTERN.finditer(text))
    if not matches:
        raise ValueError(
            "出馬表の形式を認識できませんでした。ページのレイアウトが"
            "想定と違う可能性があります。CSV貼り付けの方をお試しください。"
        )

    rows = []
    for m in matches:
        d = m.groupdict()
        rows.append({
            "horse_num": int(d["horse_num"]),
            "horse_name": d["name"].strip(),
            "waku": int(d["waku"]),
            "sex": d["sex"],
            "age": int(d["age"]),
            "jockey": d["jockey"],
            "trainer": d["stable"].strip(),
            "running_style": "差し",  # netkeibaの出馬表ページには脚質情報が無いため既定値
            "weight_carry": float(d["weight_carry"]),
            # 前日予想では馬体重がまだ発表されていないことが多いので、無ければ既定値(460/0)を使う
            "horse_weight": int(d["horse_weight"]) if d.get("horse_weight") else 460,
            "weight_diff": int(d["weight_diff"]) if d.get("weight_diff") else 0,
            "prev_rank": 0,
            "rest_weeks": 0,
            # 前日は人気・オッズが確定していないことも多いので、無ければ既定値を使う
            "popularity": int(d["popularity"]) if d.get("popularity") else int(d["horse_num"]),
            "odds": float(d["odds"]) if d.get("odds") else 10.0,
        })
    return pd.DataFrame(rows)[HORSE_TABLE_COLS]


def parse_pasted_csv(text: str) -> pd.DataFrame:
    """CSVテキストを出走馬テーブルの形式に変換する。

    列が足りない場合は妥当な既定値で埋める(prev_rank/rest_weeksは0=不明など)。
    """
    df = pd.read_csv(io.StringIO(text.strip()))
    df.columns = [c.strip() for c in df.columns]

    # 学習・表示に不要な列(race_id, venue, distance, track_type, condition, finish_rank)は無視する
    df = df[[c for c in df.columns if c in HORSE_TABLE_COLS]]

    defaults = {
        "waku": None,  # horse_numで後埋め
        "horse_name": "",
        "sex": "牡",
        "age": 4,
        "jockey": "UNK",
        "trainer": "UNK",
        "running_style": "差し",
        "weight_carry": 55.0,
        "horse_weight": 460,
        "weight_diff": 0,
        "prev_rank": 0,
        "rest_weeks": 0,
        "popularity": None,  # 後で順位を振る
        "odds": 10.0,
    }
    if "horse_num" not in df.columns:
        raise ValueError("horse_num列が見つかりません。CSVの1行目(ヘッダー)を確認してください。")

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    if df["waku"].isna().any():
        df["waku"] = df["waku"].fillna(df["horse_num"])
    if df["popularity"].isna().any():
        df["popularity"] = range(1, len(df) + 1)

    df = df[HORSE_TABLE_COLS]
    return df.reset_index(drop=True)


def get_horse_full_history(horse_name_query: str) -> pd.DataFrame:
    """指定した馬名(部分一致)の全レース成績を、時系列順に取得する。

    展開評価・レースタイム水準・相手のレベル評価・レースレベルもまとめて計算して返す。
    """
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    hist = pd.read_csv(DATA_PATH)
    if "horse_name" not in hist.columns:
        return pd.DataFrame()

    hist = hist.dropna(subset=["horse_name"])
    hist = hist[hist["horse_name"].astype(str).str.strip() != ""]
    if hist.empty:
        return pd.DataFrame()

    hist = compute_pace_note(hist)
    hist = compute_race_time_level(hist)
    hist = compute_field_strength_note(hist)
    hist = compute_stretch_out_note(hist)
    hist = add_chronological_sort_key(hist)
    hist = hist.sort_values("_chron_key")

    query_norm = _normalize_name(horse_name_query)
    name_norm = hist["horse_name"].astype(str).apply(_normalize_name)
    matched = hist[name_norm.str.contains(query_norm, na=False)]
    return matched


def _format_time(time_sec):
    """秒数を「1:10.8」のような表記に変換する。"""
    try:
        t = float(time_sec)
    except (TypeError, ValueError):
        return "不明"
    minutes = int(t // 60)
    seconds = t - minutes * 60
    if minutes > 0:
        return f"{minutes}:{seconds:04.1f}"
    return f"{seconds:.1f}"


def _x_weight(text: str) -> int:
    """Xの文字数カウント方式を概算する(全角相当の文字は2、それ以外は1として計算)。

    Xの無料枠は実質全角140文字(=280相当)までなので、この関数で概算しながら
    ツイート文を組み立てる。
    """
    import unicodedata
    total = 0
    for ch in text:
        w = unicodedata.east_asian_width(ch)
        total += 2 if w in ("W", "F") else 1
    return total


def generate_tweet_text(result: pd.DataFrame, race_label: str = "") -> str:
    """予測結果(印付きのDataFrame)から、そのままXに貼れるツイート文を作る。

    ◎○▲は馬名込み、△・⭐は馬番だけの簡潔な表記にすることで、
    頭数が多いレースでも文字数(280相当)に収まりやすくする。
    """
    lines = []
    if race_label:
        lines.append(f"【{race_label}】")

    for mark in ["◎", "○", "▲"]:
        row = result[result["印"] == mark]
        if len(row):
            r = row.iloc[0]
            lines.append(f"{mark}{r['馬番']}{r['馬名']}")

    for mark, label in [("△", "△"), ("⭐", "⭐")]:
        rows = result[result["印"] == mark]
        if len(rows):
            nums = ".".join(str(n) for n in rows["馬番"])
            lines.append(f"{label}{nums}")

    text = "\n".join(lines) + "\n\n#競馬"

    # 文字数(概算)が280相当を超えていたら、最後の行から順に削って調整する
    while _x_weight(text) > 280 and len(lines) > 1:
        lines = lines[:-1]
        text = "\n".join(lines) + "\n\n#競馬"

    return text


def generate_scouting_memos(max_n: int = 5) -> list:
    """展開評価・距離延長候補・相手のレベル評価が付いた馬について、
    そのままnoteに貼れる下書き文章(1頭1段落)のリストを生成する。

    直近(chronological順で新しい)の該当馬から並べ、max_n件まで返す。
    """
    if not os.path.exists(DATA_PATH):
        return []
    hist = pd.read_csv(DATA_PATH)
    if "horse_name" not in hist.columns:
        return []

    hist = hist.dropna(subset=["horse_name"])
    hist = hist[hist["horse_name"].astype(str).str.strip() != ""]
    if hist.empty:
        return []

    hist = compute_pace_note(hist)
    hist = compute_race_time_level(hist)
    hist = compute_field_strength_note(hist)
    hist = compute_stretch_out_note(hist)
    hist = add_chronological_sort_key(hist)
    hist = hist.sort_values("_chron_key", ascending=False)  # 新しい順

    notable = hist[
        (hist["pace_note"] != PACE_NOTE_NONE)
        | (hist["stretch_out_note"] != STRETCH_OUT_NOTE_NONE)
        | (hist["field_strength_note"] != FIELD_NOTE_NONE)
    ]
    notable = notable.head(max_n)

    drafts = []
    for _, row in notable.iterrows():
        name = row.get("horse_name", "不明")
        venue = row.get("venue", "")
        distance = row.get("distance", "")
        track_type = row.get("track_type", "")
        condition = row.get("condition", "")
        finish = row.get("finish_rank", "")
        popularity = row.get("popularity", "")
        odds = row.get("odds", "")
        time_str = _format_time(row.get("time_sec"))
        agari = row.get("agari_3f", "")

        lines = [f"■ {name}({venue}{distance}m{track_type}・{condition}/{finish}着・{popularity}番人気・オッズ{odds}倍)", ""]
        lines.append(f"タイムは{time_str}、上がり3Fは{agari}でした。")

        note_texts = []
        if row.get("pace_note") == PACE_NOTE_LEADER_GRIT:
            note_texts.append(
                "先行して上がりは平凡だったにも関わらず勝ち切っています。展開関係なく地力で押し切れる、"
                "素質を感じさせる内容です。"
            )
        elif row.get("pace_note") == PACE_NOTE_LEADER_STRONG_RACE:
            note_texts.append(
                "先行して上がりは平凡だったにも関わらず2-3着に好走しています。勝ち切れてはいませんが、"
                "展開関係なく粘れる強さを感じさせる内容です。"
            )
        elif row.get("pace_note") == "展開不利(上がり1位なのに掲示板外)":
            note_texts.append(
                "上がり3Fはこのレースで最速だったにも関わらず掲示板に載れませんでした。位置取りや展開の綾で"
                "割を食った可能性があります。"
            )
        if row.get("stretch_out_note", STRETCH_OUT_NOTE_NONE) != STRETCH_OUT_NOTE_NONE:
            note_texts.append(
                "短距離であまり追走できておらず、それでも上がりは目立って速い内容でした。"
                "距離が延びれば足が生きてくるかもしれません。次に距離延長で出てきたら注目です。"
            )
        if row.get("field_strength_note", FIELD_NOTE_NONE) != FIELD_NOTE_NONE:
            note_texts.append(
                "このレースで僅差の先着を許した馬のうち複数が、その後の別レースで掲示板(5着以内)に"
                "載っています。相手のレベルが高かった可能性があり、着順以上に評価しても良さそうです。"
            )

        lines.append(" ".join(note_texts))
        drafts.append("\n".join(lines))

    return drafts


def lookup_horse_history() -> pd.DataFrame:
    """data/dummy_races.csv(実データ)から、各馬名のレース結果を取得する。

    戻り値: (latest, full)
    - latest: horse_nameをキーにした「最新レースだけ」のDataFrame
    - full:   horse_name列を持つ「全レース分」のDataFrame(得意不得意判定用)
    """
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(), pd.DataFrame()
    hist = pd.read_csv(DATA_PATH)
    if "horse_name" not in hist.columns:
        return pd.DataFrame(), pd.DataFrame()

    hist = hist.dropna(subset=["horse_name"])
    hist = hist[hist["horse_name"].astype(str).str.strip() != ""]
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame()

    hist = compute_pace_note(hist)  # 各レース内での展開評価(強い勝ち方/展開不利)を付与
    hist = compute_race_time_level(hist)  # レースタイムの水準を付与
    hist = compute_field_strength_note(hist)  # 先着馬のその後の評価を付与
    hist = compute_stretch_out_note(hist)  # 距離延長で変わるかもの判定を付与
    hist = add_chronological_sort_key(hist)
    hist = hist.sort_values("_chron_key")
    latest = hist.groupby("horse_name", as_index=True).tail(1).set_index("horse_name")
    return latest, hist


def _typical_running_style(name_hist: pd.DataFrame):
    """ある馬の過去走の脚質のうち、一番多く見せている脚質(最頻値)を返す。

    同数で並んだ場合は、直近の走りにある方を優先する
    (chronologicalに新しい順に見て最初に出てきた方を採用)。
    データが無ければNoneを返す。
    """
    if name_hist.empty or "running_style" not in name_hist.columns:
        return None
    styles = name_hist["running_style"].dropna()
    styles = styles[styles.isin(RUNNING_STYLES)]
    if styles.empty:
        return None
    counts = styles.value_counts()
    top_count = counts.max()
    top_styles = set(counts[counts == top_count].index)
    if len(top_styles) == 1:
        return top_styles.pop()

    # 同数で並んだ場合は、時系列で新しい方から見て最初に登場した脚質を採用する
    ordered = add_chronological_sort_key(name_hist).sort_values("_chron_key", ascending=False)
    for style in ordered["running_style"]:
        if style in top_styles:
            return style
    return top_styles.pop()


def compute_current_aptitude(name_hist: pd.DataFrame, current_turn: str, current_hill: str, current_distance=None):
    """ある馬の全過去成績から、今回のレース条件(右左回り・坂・距離区分)との
    得意不得意を判定する。

    予測対象のレースはまだ走っていないので、その馬の「これまでの全成績」を使って良い
    (学習時のcompute_horse_course_aptitude/compute_horse_distance_aptitudeとは違い、
    未来のデータリークの心配は無い)。
    """
    if name_hist.empty or "venue" not in name_hist.columns or "finish_rank" not in name_hist.columns:
        return APTITUDE_UNKNOWN, APTITUDE_UNKNOWN, APTITUDE_UNKNOWN

    turns = name_hist["venue"].map(COURSE_TURN).fillna("右")
    hills = name_hist["venue"].map(COURSE_HILL).fillna("坂なし")
    top3 = (name_hist["finish_rank"] <= 3).astype(int)

    def _judge(same_mask, diff_mask, label):
        same = top3[same_mask]
        diff = top3[diff_mask]
        if len(same) < 1 or len(diff) < 1:
            return APTITUDE_UNKNOWN
        gap = same.mean() - diff.mean()
        if gap >= 0.34:
            return f"得意({label}好走歴あり)"
        if gap <= -0.34:
            return f"不得意({label}苦手傾向)"
        return "差なし"

    turn_apt = _judge(turns == current_turn, turns != current_turn, f"{current_turn}回り")
    hill_apt = _judge(hills == current_hill, hills != current_hill, current_hill)

    dist_apt = APTITUDE_UNKNOWN
    if current_distance is not None and "distance" in name_hist.columns:
        cats = name_hist["distance"].apply(distance_category)
        cur_cat = distance_category(current_distance)
        dist_apt = _judge(cats == cur_cat, cats != cur_cat, cur_cat)

    return turn_apt, hill_apt, dist_apt


def _normalize_name(name: str) -> str:
    """馬名の表記ゆれ(全角/半角スペースなど)を吸収するための正規化。"""
    return str(name).strip().replace("\u3000", "").replace(" ", "")


def apply_horse_history(
    df: pd.DataFrame, history: pd.DataFrame, full_history: pd.DataFrame,
    current_turn: str, current_hill: str, current_distance=None,
):
    """出走馬テーブルのhorse_nameを使って前走成績(prev_rank)を自動入力する。

    見つかった前走のタイム・上がり3F・通過順・展開評価や、今回のレース条件
    (右左回り・坂)との得意不得意は、表には出さず st.session_state.prev_extra に
    保存しておき、予測時にこっそり使う。
    戻り値には、見つからなかった馬名と近い候補(不一致デバッグ用)も含める。
    """
    import difflib

    df = df.copy()
    extra = {}
    matched = 0
    unmatched_suggestions = {}

    # 正規化した馬名 -> 実際のインデックス名、のマップを作る(表記ゆれ対策)
    norm_to_actual = {_normalize_name(idx): idx for idx in history.index}

    for i, row in df.iterrows():
        raw_name = str(row.get("horse_name", "")).strip()
        if not raw_name:
            continue
        norm_name = _normalize_name(raw_name)

        actual_key = norm_to_actual.get(norm_name)
        if actual_key is None:
            # 近い候補があれば提示する(スペルミスや表記ゆれの発見用)
            close = difflib.get_close_matches(norm_name, norm_to_actual.keys(), n=1, cutoff=0.6)
            if close:
                unmatched_suggestions[raw_name] = norm_to_actual[close[0]]
            continue

        h = history.loc[actual_key]
        if isinstance(h, pd.DataFrame):  # 同名馬が複数いる場合は最初の1件
            h = h.iloc[0]
        if pd.notna(h.get("finish_rank")):
            df.at[i, "prev_rank"] = int(h["finish_rank"])
            matched += 1

        # この馬の全過去成績(複数走あれば全部)を先に取得しておく
        name_hist = pd.DataFrame()
        if not full_history.empty and "horse_name" in full_history.columns:
            name_hist = full_history[full_history["horse_name"] == actual_key]

        # 出馬表には脚質情報が無いため既定値(差し)になっているが、
        # 過去走の脚質が分かるならそちらを引き継いだ方が精度が上がる。
        # 複数走あれば「一番多く見せている脚質」を採用する(1走だけだと展開に
        # 左右されたブレの可能性があるため。逃げ→逃げ→差し のような馬なら
        # 「逃げ」を本来の脚質とみなす、という考え方)。
        typical_style = _typical_running_style(name_hist) if not name_hist.empty else None
        if typical_style is None:
            typical_style = h.get("running_style")
        if pd.notna(typical_style) and typical_style in RUNNING_STYLES and "running_style" in df.columns:
            df.at[i, "running_style"] = typical_style

        # この馬の全過去成績から、今回のコース条件・距離との得意不得意を判定
        turn_apt, hill_apt, dist_apt = APTITUDE_UNKNOWN, APTITUDE_UNKNOWN, APTITUDE_UNKNOWN
        if not name_hist.empty:
            turn_apt, hill_apt, dist_apt = compute_current_aptitude(
                name_hist, current_turn, current_hill, current_distance
            )

        extra[raw_name] = {
            "prev_time_sec": h.get("time_sec", 0) if pd.notna(h.get("time_sec")) else 0,
            "prev_agari_3f": h.get("agari_3f", 0) if pd.notna(h.get("agari_3f")) else 0,
            "prev_corner_pos": h.get("corner_pos", 0) if pd.notna(h.get("corner_pos")) else 0,
            "prev_pace_note": h.get("pace_note", PACE_NOTE_NONE) if pd.notna(h.get("pace_note")) else PACE_NOTE_NONE,
            "prev_class_level": h.get("class_level", 0) if pd.notna(h.get("class_level")) else 0,
            "prev_race_time_score": h.get("race_time_score", 0) if pd.notna(h.get("race_time_score")) else 0,
            "prev_field_strength_note": h.get("field_strength_note", FIELD_NOTE_NONE) if pd.notna(h.get("field_strength_note")) else FIELD_NOTE_NONE,
            "prev_stretch_out_note": h.get("stretch_out_note", STRETCH_OUT_NOTE_NONE) if pd.notna(h.get("stretch_out_note")) else STRETCH_OUT_NOTE_NONE,
            "horse_turn_aptitude": turn_apt,
            "horse_hill_aptitude": hill_apt,
            "horse_distance_aptitude": dist_apt,
        }
    return df, matched, extra, unmatched_suggestions


MARKS = ["◎", "○", "▲", "△", "△"]


def assign_marks(n: int) -> list:
    """上位から ◎ ○ ▲ △ △ の印を割り当て、残りは無印にする。

    出走頭数が少ない場合は、印の数を頭数に応じて減らす。
    """
    if n <= 3:
        marks = MARKS[:n]
    elif n <= 6:
        marks = MARKS[:3]
    elif n <= 9:
        marks = MARKS[:4]
    else:
        marks = MARKS[:5]
    return marks + [""] * (n - len(marks))


def compute_value_gap(ai_rank: pd.Series, popularity: pd.Series) -> pd.Series:
    """AIの評価順位と市場(人気)のズレを計算する。

    プラスが大きいほど「AIは人気より高く評価している」= 妙味のある穴馬候補。
    マイナスが大きいほど「人気ほどには評価していない」= 過剰人気の疑い。
    """
    return (popularity.astype(float) - ai_rank.astype(float)).round(0).astype(int)


def gap_label(gap: int) -> str:
    if gap >= 4:
        return "妙味大"
    if gap >= 2:
        return "妙味あり"
    if gap <= -4:
        return "過剰人気"
    if gap <= -2:
        return "やや人気先行"
    return "妥当"


@st.cache_resource
def load_model(_feature_hash: str):
    """毎回その場で学習してモデルを作る(保存済みmodel.joblibは使わない)。

    引数の_feature_hashは、特徴量セット(FEATURE_COLS)が変わった時に
    自動でキャッシュを無効化するための仕組み(値そのものは使わない)。
    ファイルに保存したモデルを使い回す方式だと、列構成を変更した時に
    古いモデルとズレて壊れる(KeyErrorなど)ため、st.cache_resourceの
    キャッシュだけに頼る。
    データCSV(data/dummy_races.csv)は、既に置き換えた実データを
    誤って上書きしないよう、無い時だけダミーデータを生成する。
    """
    from data.generate_dummy_data import generate
    from model.train_model import train

    needs_regen = True
    if os.path.exists(DATA_PATH):
        existing_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
        # 「CSVファイルに実際に保存されているべき生の列」だけで判定する。
        # FEATURE_COLS全体で判定すると、prev_time_secなどその場で計算される列が
        # 常に「無い」と判定されてしまい、実データがダミーデータで上書きされる
        # 重大なバグになるため、RAW_REQUIRED_COLSを使う。
        needs_regen = not set(RAW_REQUIRED_COLS).issubset(set(existing_cols))

    if needs_regen:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        generate(DATA_PATH, verbose=False)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    return train(DATA_PATH, MODEL_PATH, verbose=False)


def main():
    st.set_page_config(page_title="KEIBA AI｜競馬予想", page_icon="🏇", layout="wide")

    st.markdown(
        """
        <style>
        #MainMenu, footer, header {visibility: hidden;}

        .hero {
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #0F1B2E 0%, #1A3A52 100%);
            color: #F5F5F5;
            margin-bottom: 1.6rem;
            border-left: 6px solid #D4AF37;
            box-shadow: 0 4px 12px rgba(212, 175, 55, 0.15);
        }
        .hero h1 {
            margin: 0;
            font-size: 1.9rem;
            letter-spacing: 0.02em;
            color: #D4AF37;
        }
        .hero p {
            margin: 0.3rem 0 0 0;
            color: #C0C0C0;
            font-size: 0.95rem;
        }
        .section-head {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-left: 5px solid #D4AF37;
            padding-left: 0.6rem;
            margin: 1.4rem 0 0.6rem 0;
        }
        .section-head .num {
            background: linear-gradient(135deg, #D4AF37 0%, #B8860B 100%);
            color: #0F1B2E;
            border-radius: 50%;
            width: 1.6rem;
            height: 1.6rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 700;
            box-shadow: 0 2px 6px rgba(212, 175, 55, 0.3);
        }
        .section-head .label {
            font-size: 1.05rem;
            font-weight: 600;
            color: #F5F5F5;
        }
        .disclaimer {
            font-size: 0.78rem;
            color: #999999;
            border-top: 1px solid #D4AF37;
            border-top-opacity: 0.3;
            padding-top: 0.6rem;
            margin-top: 1rem;
        }
        
        /* 印の装飾（◎○▲△⭐） */
        .mark-symbol {
            display: inline-block;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-weight: 700;
            font-size: 1.1rem;
            margin-right: 0.3rem;
        }
        .mark-ace {
            color: #D4AF37;
            text-shadow: 0 0 8px rgba(212, 175, 55, 0.5);
        }
        .mark-big {
            color: #FFD700;
        }
        .mark-win {
            color: #FFB6C1;
        }
        .mark-horse {
            color: #87CEEB;
        }
        .mark-star {
            color: #FFD700;
            filter: drop-shadow(0 0 4px rgba(255, 215, 0, 0.6));
        }
        </style>
        <div class="hero">
            <h1>🏇 KEIBA AI</h1>
            <p>過去のレースデータから、出走馬の複勝圏内(3着以内)の可能性をAIが評価します。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def section_head(num: str, label: str):
        st.markdown(
            f'<div class="section-head"><span class="num">{num}</span>'
            f'<span class="label">{label}</span></div>',
            unsafe_allow_html=True,
        )

    with st.spinner("モデルを準備しています..."):
        bundle = load_model(FEATURE_HASH)

    if "collect_paste_version" not in st.session_state:
        st.session_state.collect_paste_version = 0
    if "result_urls_version" not in st.session_state:
        st.session_state.result_urls_version = 0
    if "shutuba_url_version" not in st.session_state:
        st.session_state.shutuba_url_version = 0

    with st.expander("📥 データを集める(結果ページを貼り付けて自動保存)", expanded=True):
        st.caption(
            "netkeibaの「結果」ページの表をコピーして貼り付けると、解析してGitHub上の"
            "data/dummy_races.csvに追記できます。**複数レース分をまとめて貼り付けてもOK**です"
            "(「発走」を含む行の数だけレースを自動で見つけます)。"
        )

        colA, colB = st.columns([2, 1])
        with colA:
            use_auto_id = st.checkbox("race_idを自動で決める(推奨)", value=True)
        with colB:
            manual_start_id = st.number_input(
                "開始race_id", min_value=1, value=9001, step=1, disabled=use_auto_id,
            )

        collect_date = st.date_input(
            "このレース(複数貼り付けた場合は全部)の開催日",
            help=(
                "前走の判定を「追加した順番」ではなく「実際の日付」で行うために使います。"
                "後から昔のレースを追加しても、前走が正しく判定されるようになります。"
            ),
        )

        with st.expander("🔗 URLを貼るだけで取得する(実験的機能)"):
            st.caption(
                "コピペの代わりに、netkeibaの結果ページのURLを貼って自動取得できます。"
                "複数レース分まとめて取得したい場合は、1行に1URLずつ貼ってください。"
                "うまく取れない場合は上のコピペ方式を使ってください。"
            )
            result_urls_text = st.text_area(
                "結果ページのURL(1行に1つ)", height=100,
                key=f"result_urls_input_{st.session_state.result_urls_version}",
            )
            if st.button("URLから取得してプレビュー", key="result_url_fetch"):
                urls = [u.strip() for u in result_urls_text.splitlines() if u.strip()]
                if not urls:
                    st.warning("URLを1つ以上入力してください。")
                else:
                    try:
                        token = st.secrets["github_token"]
                        repo = st.secrets["github_repo"]
                        branch = st.secrets.get("github_branch", "main")
                        existing_df_u, _ = fetch_csv(token, repo, branch, "data/dummy_races.csv")

                        if use_auto_id:
                            next_id_u = (int(existing_df_u["race_id"].max()) + 1) if len(existing_df_u) else 9001
                        else:
                            next_id_u = int(manual_start_id)

                        all_dfs = []
                        overwritten_list = []
                        failed_urls = []
                        progress = st.progress(0.0, text="取得中...")
                        for i, url in enumerate(urls):
                            progress.progress((i + 1) / len(urls), text=f"取得中... ({i + 1}/{len(urls)})")
                            try:
                                one_df = fetch_and_parse_netkeiba_result(
                                    url, race_id=next_id_u, race_date=str(collect_date),
                                )
                            except Exception as e:
                                failed_urls.append((url, str(e)))
                                continue

                            # 既存データ・今回すでに取得した分と重複していないか確認する
                            combined_existing = pd.concat([existing_df_u] + all_dfs, ignore_index=True) if all_dfs else existing_df_u
                            race_row_u = one_df.iloc[0]
                            race_info_u = [{
                                "venue": race_row_u["venue"], "distance": race_row_u["distance"],
                                "track_type": race_row_u["track_type"], "race_date": race_row_u.get("race_date"),
                                "horse_names": one_df["horse_name"].dropna().astype(str).tolist(),
                            }]
                            matches_u = find_existing_race_ids(combined_existing, race_info_u)
                            if 0 in matches_u:
                                one_df["race_id"] = matches_u[0]
                                overwritten_list.append(matches_u[0])
                            else:
                                next_id_u += 1

                            all_dfs.append(one_df)
                        progress.empty()

                        if not all_dfs:
                            st.error("どのURLからも取得できませんでした。")
                            for url, err in failed_urls:
                                st.caption(f"・{url} → {err}")
                        else:
                            st.session_state.collect_preview = pd.concat(all_dfs, ignore_index=True)
                            n_horses = sum(len(d) for d in all_dfs)
                            msg = f"{len(all_dfs)}レース・{n_horses}頭分を取得しました。"
                            if overwritten_list:
                                msg += f" うち{len(overwritten_list)}レースは上書き対象です(race_id: {overwritten_list})。"
                            st.toast(msg, icon="✅")
                            if failed_urls:
                                st.warning(f"{len(failed_urls)}件のURLは取得できませんでした。")
                                for url, err in failed_urls:
                                    st.caption(f"・{url} → {err}")
                            st.session_state.result_urls_version += 1
                            st.rerun()
                    except KeyError:
                        st.error(
                            "GitHubのトークンが設定されていないため、重複チェックができません。"
                            "Streamlit CloudのSecretsにgithub_token / github_repoを設定してください。"
                        )
                    except Exception as e:
                        st.error(f"取得・解析に失敗しました: {e}")

        collect_text = st.text_area(
            "netkeibaの結果ページ(複数レース分をまとめて貼り付けてもOK)",
            height=200, key=f"collect_paste_{st.session_state.collect_paste_version}",
        )

        if st.button("プレビュー"):
            try:
                token = st.secrets["github_token"]
                repo = st.secrets["github_repo"]
                branch = st.secrets.get("github_branch", "main")
                existing_df, _ = fetch_csv(token, repo, branch, "data/dummy_races.csv")

                if use_auto_id:
                    start_id = (int(existing_df["race_id"].max()) + 1) if len(existing_df) else 9001
                else:
                    start_id = int(manual_start_id)

                preview_df = parse_netkeiba_results_multi(
                    collect_text, start_race_id=start_id, race_date=str(collect_date),
                )

                # 既存データと(開催場・距離・コース種別・開催日、または出走馬名の重なり)
                # が一致するレースがあれば、新しいrace_idではなく既存のrace_idを使う(=上書きになる)
                races_info = []
                for rid in sorted(preview_df["race_id"].unique()):
                    race_rows = preview_df[preview_df["race_id"] == rid]
                    row = race_rows.iloc[0]
                    races_info.append({
                        "venue": row["venue"], "distance": row["distance"],
                        "track_type": row["track_type"], "race_date": row.get("race_date"),
                        "horse_names": race_rows["horse_name"].dropna().astype(str).tolist(),
                    })
                matches = find_existing_race_ids(existing_df, races_info)

                overwritten = []
                for i, rid in enumerate(sorted(preview_df["race_id"].unique())):
                    if i in matches:
                        preview_df.loc[preview_df["race_id"] == rid, "race_id"] = matches[i]
                        overwritten.append(matches[i])

                st.session_state.collect_preview = preview_df
                n_races = preview_df["race_id"].nunique()
                msg = f"{n_races}レース・{len(preview_df)}頭分を解析しました。"
                if overwritten:
                    msg += f" うち{len(overwritten)}レースは既存データと同じ内容のため、上書き対象になります(race_id: {overwritten})。"
                st.success(msg + " 内容を確認してから保存してください。")
            except KeyError:
                st.error(
                    "GitHubのトークンが設定されていないため、重複チェックができません。"
                    "Streamlit CloudのSecretsにgithub_token / github_repoを設定してください。"
                )
            except Exception as e:
                st.error(str(e))

        if "collect_preview" in st.session_state:
            st.dataframe(st.session_state.collect_preview, use_container_width=True)

            with st.expander("🔄 コーナー通過順位を後から貼り付けて反映する(行内に無かった場合)"):
                st.caption(
                    "netkeibaページ下部の「コーナー通過順位」の部分"
                    "(例: 「3コーナー\\t(*1,3)(7,8)...」のような行)をまとめて貼り付けてください。"
                )
                preview_race_ids = sorted(st.session_state.collect_preview["race_id"].unique())
                target_race_id = st.selectbox("対象のrace_id", preview_race_ids, key="corner_target_race")
                corner_text = st.text_area(
                    "コーナー通過順位セクション",
                    height=100,
                    key=f"corner_paste_{st.session_state.get('corner_paste_version', 0)}",
                )
                if st.button("反映する", key="corner_apply"):
                    apply_ok = False
                    try:
                        st.session_state.collect_preview = apply_corner_section_to_df(
                            st.session_state.collect_preview, corner_text, int(target_race_id),
                        )
                        st.toast("コーナー通過順位を反映しました。", icon="✅")
                        st.session_state.corner_paste_version = st.session_state.get("corner_paste_version", 0) + 1
                        apply_ok = True
                    except Exception as e:
                        st.error(str(e))
                    if apply_ok:
                        st.rerun()

            if st.button("✅ GitHubに保存する", type="primary"):
                save_succeeded = False
                try:
                    token = st.secrets["github_token"]
                    repo = st.secrets["github_repo"]
                    branch = st.secrets.get("github_branch", "main")
                    n_races = st.session_state.collect_preview["race_id"].nunique()
                    total = append_rows_to_csv(
                        token, repo, branch, "data/dummy_races.csv",
                        st.session_state.collect_preview,
                        message=f"Add {n_races} race(s) data",
                    )
                    st.toast(f"GitHubに保存しました!(CSV全体: {total}行)数分後にアプリが再デプロイされます。", icon="✅")
                    del st.session_state.collect_preview
                    st.session_state.collect_paste_version += 1
                    save_succeeded = True
                except KeyError:
                    st.error(
                        "GitHubのトークンが設定されていません。Streamlit Cloudの「Manage app」→"
                        "「Settings」→「Secrets」に github_token / github_repo を設定してください。"
                    )
                except Exception as e:
                    st.error(f"保存に失敗しました: {e}")

                # st.rerun()はtry/exceptの外で呼ぶ(内部的な例外がexceptに
                # 誤って捕まり、成功時にもエラー表示が出てしまうのを防ぐため)
                if save_succeeded:
                    st.rerun()

    with st.expander("📊 的中率を確認する(バックテスト)"):
        st.caption(
            "今まで貯まったデータを使い、「そのレースを学習に使わずに予測する」形で"
            "印(◎○▲△)ごとの複勝率・勝率を検証します。データが増えるほど信頼できる結果になります。"
        )
        age_filter = st.selectbox(
            "対象馬齢",
            ["全馬", "2歳馬のみ", "3歳以上のみ"],
            help="他の馬齢のデータを混ぜても、特定の馬齢の精度が落ちていないか確認したい時に使ってください。",
        )
        if st.button("バックテストを実行"):
            with st.spinner("検証中です(データ量によって数秒〜数十秒かかります)..."):
                try:
                    st.session_state.backtest_result = backtest(DATA_PATH, age_filter=age_filter)
                except Exception as e:
                    st.error(f"検証に失敗しました: {e}")

        if "backtest_result" in st.session_state:
            r = st.session_state.backtest_result
            st.caption(f"対象: {r['n_races']}レース・{r['n_horses']}頭")

            rows = []
            for mark, stats in r["mark_stats"].items():
                rows.append({
                    "印": mark, "件数": stats["件数"],
                    "複勝率": f"{stats['複勝率']*100:.1f}%",
                    "勝率": f"{stats['勝率']*100:.1f}%",
                })
            if r["favorite_stats"]:
                fs = r["favorite_stats"]
                rows.append({
                    "印": "(参考)1番人気", "件数": fs["件数"],
                    "複勝率": f"{fs['複勝率']*100:.1f}%",
                    "勝率": f"{fs['勝率']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "「(参考)1番人気」は、毎回オッズ1番人気の馬を買った場合の成績です。"
                "◎の成績がこれを上回っていれば、AIの予測が市場の評価に対して"
                "何らかの上乗せ価値を出せている、という目安になります。"
            )

    with st.expander("🔎 馬名でデータを検索する(データ蓄積の確認・振り返り用)"):
        st.caption(
            "馬名を入力すると、その馬の全レース成績と、うちのAIが見ている中身"
            "(展開評価・レースレベル・相手の強さ・距離延長候補かどうか)を確認できます。"
        )
        search_name = st.text_input("馬名(部分一致でOK)", key="horse_search_input")
        if st.button("検索", key="horse_search_btn"):
            if not search_name.strip():
                st.warning("馬名を入力してください。")
            else:
                found = get_horse_full_history(search_name)
                if found.empty:
                    st.info(f"「{search_name}」に一致するデータが見つかりませんでした。")
                else:
                    st.session_state.horse_search_result = found

        if "horse_search_result" in st.session_state:
            found = st.session_state.horse_search_result
            for name in found["horse_name"].unique():
                sub = found[found["horse_name"] == name]
                st.markdown(f"**{name}**({len(sub)}走)")
                rows = []
                for _, row in sub.iterrows():
                    level = describe_race_level(row.get("race_class", ""), row.get("race_time_score", 0))
                    notes = []
                    if row.get("pace_note", PACE_NOTE_NONE) != PACE_NOTE_NONE:
                        notes.append(row["pace_note"])
                    if row.get("field_strength_note", FIELD_NOTE_NONE) != FIELD_NOTE_NONE:
                        notes.append(row["field_strength_note"])
                    if row.get("stretch_out_note", STRETCH_OUT_NOTE_NONE) != STRETCH_OUT_NOTE_NONE:
                        notes.append(row["stretch_out_note"])
                    rows.append({
                        "開催日": row.get("race_date") or "不明",
                        "レース": f"{row['venue']}{row['distance']}m{row['track_type']}",
                        "着順": row["finish_rank"],
                        "人気": row.get("popularity", ""),
                        "オッズ": row.get("odds", ""),
                        "レースレベル": level,
                        "AIの着眼点": " / ".join(notes) if notes else "特になし",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("📝 注目馬メモを自動生成する(note下書き)"):
        st.caption(
            "展開評価・距離延長候補・相手のレベル評価が付いた馬を自動でリストアップし、"
            "そのままnoteに貼れる下書き文章を作ります(直近の該当馬から新しい順)。"
        )
        max_n = st.slider("生成する件数(上限)", 1, 20, 5, key="scouting_memo_n")
        if st.button("下書きを生成する", key="scouting_memo_btn"):
            drafts = generate_scouting_memos(max_n=max_n)
            if not drafts:
                st.info("現時点で該当する馬が見つかりませんでした。")
            else:
                st.session_state.scouting_memo_drafts = drafts

        if "scouting_memo_drafts" in st.session_state:
            full_text = "\n\n".join(st.session_state.scouting_memo_drafts)
            st.text_area("下書き(コピーしてnoteに貼ってください)", value=full_text, height=400)

    with st.expander("⏪ 特定のコミット時点に巻き戻す(緊急用)"):
        st.caption(
            "自動修復などで想定外にデータが壊れてしまった時、指定したコミットの"
            "時点の状態にdata/dummy_races.csvを戻します。取り消しはできないので、"
            "コミットSHA(GitHubの「Commits」履歴に出ている7〜8桁の英数字)を"
            "よく確認してから実行してください。"
        )
        revert_sha = st.text_input(
            "戻したいコミットのSHA(例: 796dce2)", key="revert_sha_input",
        )
        if st.button("このコミット時点のデータを確認する", key="revert_check_btn"):
            if not revert_sha.strip():
                st.error("コミットSHAを入力してください。")
            else:
                try:
                    token = st.secrets["github_token"]
                    repo = st.secrets["github_repo"]
                    csv_path = st.secrets.get("github_csv_path", "data/dummy_races.csv")
                    with st.spinner("取得中..."):
                        df_old, _ = fetch_csv(token, repo, revert_sha.strip(), csv_path)
                    st.session_state.revert_preview = {"df": df_old, "sha_ref": revert_sha.strip()}
                    st.toast(f"{len(df_old)}行を取得しました", icon="✅")
                except Exception as e:
                    st.error(f"取得に失敗しました(SHAが正しいか確認してください): {e}")

        if "revert_preview" in st.session_state:
            rp = st.session_state.revert_preview
            st.write(f"コミット `{rp['sha_ref']}` 時点の行数: {len(rp['df'])}行")
            st.dataframe(rp["df"].tail(10), height=200)
            st.warning("⚠️ この内容で今のGitHubのファイルを上書きします。この操作は取り消せません。")
            if st.button("この状態に巻き戻す(GitHubに反映)", key="revert_apply_btn"):
                try:
                    token = st.secrets["github_token"]
                    repo = st.secrets["github_repo"]
                    branch = st.secrets.get("github_branch", "main")
                    csv_path = st.secrets.get("github_csv_path", "data/dummy_races.csv")
                    # 現在のsha(上書き対象)を取得
                    _, current_sha = fetch_csv(token, repo, branch, csv_path)
                    update_csv(
                        token, repo, branch, csv_path, rp["df"], current_sha,
                        message=f"緊急巻き戻し(コミット{rp['sha_ref']}の状態に復元)",
                    )
                    st.success("巻き戻しました。数分後にアプリが自動で再起動します。")
                    del st.session_state.revert_preview
                except Exception as e:
                    st.error(f"反映に失敗しました: {e}")

    with st.expander("🧹 データの健康診断(重複チェック・修復)"):
        st.caption(
            "GitHub上のdata/dummy_races.csvを直接確認し、完全に同じ内容の行が"
            "重複していないかチェックします。パソコンやターミナルは不要です。"
        )
        if st.button("重複をチェックする", key="dup_check_btn"):
            try:
                token = st.secrets["github_token"]
                repo = st.secrets["github_repo"]
                branch = st.secrets.get("github_branch", "main")
                csv_path = st.secrets.get("github_csv_path", "data/dummy_races.csv")
                with st.spinner("GitHubから取得中..."):
                    df_check, sha = fetch_csv(token, repo, branch, csv_path)

                before = len(df_check)
                df_dedup = df_check.drop_duplicates(keep="first").reset_index(drop=True)
                removed = before - len(df_dedup)

                # 完全重複を除いた後も、同じrace_idに1着馬が複数いないか確認
                remaining = []
                if "finish_rank" in df_dedup.columns:
                    for rid, group in df_dedup.groupby("race_id"):
                        n_winners = (group["finish_rank"] == 1).sum()
                        if n_winners >= 2:
                            remaining.append((rid, int(n_winners)))

                # 別のrace_idなのに、出走馬がほぼ同じ(=同じレースを2回登録した疑い)を検出
                same_race_pairs = []
                if "horse_name" in df_dedup.columns:
                    race_horses = df_dedup.groupby("race_id")["horse_name"].apply(
                        lambda s: frozenset(s.dropna())
                    ).to_dict()
                    seen_ids = list(race_horses.items())
                    for i, (rid, horses) in enumerate(seen_ids):
                        if not horses:
                            continue
                        for other_rid, other_horses in seen_ids[:i]:
                            if not other_horses:
                                continue
                            overlap = horses & other_horses
                            smaller = min(len(horses), len(other_horses))
                            if smaller > 0 and len(overlap) / smaller >= 0.8 and abs(len(horses) - len(other_horses)) <= 2:
                                same_race_pairs.append((rid, other_rid))

                st.session_state.dup_check_result = {
                    "before": before, "after": len(df_dedup), "removed": removed,
                    "remaining": remaining, "df_dedup": df_dedup, "sha": sha,
                    "same_race_pairs": same_race_pairs,
                }
                st.toast("チェック完了", icon="✅")
            except Exception as e:
                st.error(f"チェックに失敗しました: {e}")

        if "dup_check_result" in st.session_state:
            r = st.session_state.dup_check_result
            st.write(f"元の行数: {r['before']}行 → 完全重複を除いた行数: {r['after']}行(削除: {r['removed']}行)")
            if r["remaining"]:
                st.warning(
                    f"⚠️ 完全重複の削除だけでは解決しない、race_id衝突の疑いが{len(r['remaining'])}件残っています: "
                    f"{r['remaining'][:10]}{'...' if len(r['remaining']) > 10 else ''}"
                )
                st.caption("これらは個別確認が必要です。ひとまず完全重複の削除だけ反映することもできます。")
            else:
                st.success("✅ 完全重複を削除すれば、他の問題は残っていません。")

            if r["removed"] > 0:
                if st.button("この修復をGitHubに反映する", key="dup_fix_apply"):
                    try:
                        token = st.secrets["github_token"]
                        repo = st.secrets["github_repo"]
                        branch = st.secrets.get("github_branch", "main")
                        csv_path = st.secrets.get("github_csv_path", "data/dummy_races.csv")
                        update_csv(
                            token, repo, branch, csv_path, r["df_dedup"], r["sha"],
                            message=f"重複データの自動修復({r['removed']}行削除)",
                        )
                        st.success("GitHubに反映しました。数分後にアプリが自動で再起動します。")
                        del st.session_state.dup_check_result
                    except Exception as e:
                        st.error(f"反映に失敗しました: {e}")
            elif r["removed"] == 0:
                st.info("完全重複は見つかりませんでした。反映の必要はありません。")

            if r["remaining"]:
                st.markdown("---")
                st.write("**race_id衝突の自動振り分け**")
                st.caption(
                    "同じrace_idの中で、行の並び順のまま見ていき、「1着(finish_rank=1)」が"
                    "出てくるたびに「別レースの始まり」と判断して、2つ目以降のレースには"
                    "新しいrace_idを割り振ります。"
                )
                if st.button("race_id衝突を自動で振り分ける", key="collision_split_btn"):
                    df_split = r["df_dedup"].copy()
                    remaining_ids = {rid for rid, _ in r["remaining"]}
                    next_id = int(df_split["race_id"].max()) + 1
                    split_log = []

                    new_race_id_col = df_split["race_id"].copy()
                    for rid in remaining_ids:
                        idxs = df_split.index[df_split["race_id"] == rid].tolist()
                        current_id = rid
                        seen_first_winner = False
                        for idx in idxs:
                            is_winner = df_split.loc[idx, "finish_rank"] == 1
                            if is_winner:
                                if seen_first_winner:
                                    current_id = next_id
                                    next_id += 1
                                    split_log.append((rid, current_id))
                                seen_first_winner = True
                            new_race_id_col.loc[idx] = current_id

                    df_split["race_id"] = new_race_id_col
                    st.session_state.collision_split_result = {
                        "df_split": df_split, "sha": r["sha"], "split_log": split_log,
                    }
                    st.toast(f"{len(split_log)}件の新しいrace_idを割り振りました", icon="✅")

            if "collision_split_result" in st.session_state:
                cr = st.session_state.collision_split_result
                st.write(f"新しく割り振ったrace_id: {len(cr['split_log'])}件")
                st.dataframe(
                    pd.DataFrame(cr["split_log"], columns=["元のrace_id", "新しいrace_id"]),
                    height=200,
                )
                if st.button("振り分け結果をGitHubに反映する", key="collision_split_apply"):
                    try:
                        token = st.secrets["github_token"]
                        repo = st.secrets["github_repo"]
                        branch = st.secrets.get("github_branch", "main")
                        csv_path = st.secrets.get("github_csv_path", "data/dummy_races.csv")
                        update_csv(
                            token, repo, branch, csv_path, cr["df_split"], cr["sha"],
                            message=f"race_id衝突の自動振り分け({len(cr['split_log'])}件)",
                        )
                        st.success("GitHubに反映しました。数分後にアプリが自動で再起動します。")
                        del st.session_state.collision_split_result
                        del st.session_state.dup_check_result
                    except Exception as e:
                        st.error(f"反映に失敗しました: {e}")

            if r.get("same_race_pairs"):
                st.markdown("---")
                st.write(f"**別IDなのに同じレースの疑い({len(r['same_race_pairs'])}件)**")
                st.caption("どちらを残すか、内容を見て選んでください。")
                df_ref = r["df_dedup"]
                for pair_i, (rid_a, rid_b) in enumerate(r["same_race_pairs"]):
                    st.markdown(f"race_id **{rid_a}** と race_id **{rid_b}**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption(f"race_id {rid_a}")
                        st.dataframe(
                            df_ref[df_ref["race_id"] == rid_a][["horse_name", "finish_rank", "race_date", "venue"]],
                            height=150,
                        )
                    with col_b:
                        st.caption(f"race_id {rid_b}")
                        st.dataframe(
                            df_ref[df_ref["race_id"] == rid_b][["horse_name", "finish_rank", "race_date", "venue"]],
                            height=150,
                        )
                    choice = st.radio(
                        "残す方",
                        [f"race_id {rid_a}を残す(race_id {rid_b}を削除)",
                         f"race_id {rid_b}を残す(race_id {rid_a}を削除)",
                         "どちらも残す(判断しない)"],
                        key=f"same_race_choice_{pair_i}", index=2,
                    )
                    st.session_state[f"same_race_decision_{pair_i}"] = (rid_a, rid_b, choice)
                    st.markdown("---")

                if st.button("選んだ内容をGitHubに反映する", key="same_race_apply"):
                    try:
                        df_final = r["df_dedup"].copy()
                        to_delete = []
                        for pair_i in range(len(r["same_race_pairs"])):
                            rid_a, rid_b, choice = st.session_state[f"same_race_decision_{pair_i}"]
                            if choice.startswith(f"race_id {rid_a}を残す"):
                                to_delete.append(rid_b)
                            elif choice.startswith(f"race_id {rid_b}を残す"):
                                to_delete.append(rid_a)
                        if to_delete:
                            df_final = df_final[~df_final["race_id"].isin(to_delete)].reset_index(drop=True)
                            token = st.secrets["github_token"]
                            repo = st.secrets["github_repo"]
                            branch = st.secrets.get("github_branch", "main")
                            csv_path = st.secrets.get("github_csv_path", "data/dummy_races.csv")
                            update_csv(
                                token, repo, branch, csv_path, df_final, r["sha"],
                                message=f"重複レースの削除(race_id: {to_delete})",
                            )
                            st.success(f"race_id {to_delete} を削除してGitHubに反映しました。")
                            del st.session_state.dup_check_result
                        else:
                            st.info("「どちらも残す」以外を選んだ組み合わせがありませんでした。")
                    except Exception as e:
                        st.error(f"反映に失敗しました: {e}")

    section_head("1", "レース条件")
    col0, col1, col2, col3 = st.columns(4)
    with col0:
        venue = st.selectbox("開催場", VENUES)
    with col1:
        track_type = st.selectbox("コース種別", ["芝", "ダート"])
    with col2:
        available_distances = COURSE_DISTANCES[venue][track_type]
        distance = st.selectbox("距離(m)", available_distances)
    with col3:
        condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])

    straight_length = COURSE_STRAIGHT_LENGTH[venue]
    turn_direction = COURSE_TURN[venue]
    hill = COURSE_HILL[venue]

    RACE_CLASS_OPTIONS = ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "L(リステッド)", "G3", "G2", "G1"]
    RACE_CLASS_LEVELS = {"新馬": 0, "未勝利": 0, "1勝クラス": 1, "2勝クラス": 2, "3勝クラス": 3, "オープン": 4, "L(リステッド)": 4.5, "G3": 5, "G2": 6, "G1": 7}
    col4, col5 = st.columns(2)
    with col4:
        race_class = st.selectbox("レース格", RACE_CLASS_OPTIONS, index=1)
        class_level = RACE_CLASS_LEVELS[race_class]
    with col5:
        day_bias = st.selectbox(
            "今日の馬場傾向",
            DAY_BIAS_OPTIONS,
            help="レース当日の実況・データを見て、内外や脚質の有利不利があれば選んでください。分からなければ「フラット」でOKです。",
        )
    st.caption(
        f"📏 {venue}: 直線{straight_length} / {turn_direction}回り / {hill}"
        "(固定情報として自動反映されます)"
    )
    if is_gate_sensitive_course(venue, distance, track_type):
        st.info(
            "⚠️ このコースは、他コースより枠番(内枠/外枠)の有利不利が特に強く出ることで"
            "知られています。AIの判断材料にも反映していますが、当日の馬場傾向(内有利/外有利)も"
            "できるだけ正確に選んでください。"
        )

    ignore_pace = st.checkbox(
        "脚質不問モードにする(データが少ない馬齢戦・クラス向け)",
        help=(
            "2歳新馬戦以外など、まだ馬ごとのデータが少ない条件では、脚質の自動推定(逃げ/先行/差し/追い込み)が"
            "あまり当てにならないことがあります。チェックすると、想定ペース・ペース相性の判定をニュートラルにして、"
            "脚質の推定ミスが予測に影響しにくくなります。"
        ),
    )

    min_proba_for_mark = st.slider(
        "印をつける最低ライン(複勝確率%)", min_value=0, max_value=40, value=15, step=1,
        help=(
            "複勝確率がこの数値を下回る馬は、単勝期待値がどれだけ高くても印の対象から外します。"
            "極端な人気薄は、勝率の推定自体がまだ不安定なことが多く、期待値の数字だけが偶然大きく"
            "出てしまうことがあるための足切りです。0にすると足切りなしになります。"
        ),
    )

    col_line1, col_line2 = st.columns(2)
    with col_line1:
        aite_line = st.slider(
            "相手ライン(複勝確率%・△の追加基準)", min_value=0, max_value=60, value=20, step=1,
            help="◎○▲(単勝期待値トップ1頭+複勝確率上位2頭)以外で、複勝確率がこの数値を超えた馬は全員△にします。頭数は固定しません。",
        )
    with col_line2:
        ana_line = st.slider(
            "穴ライン(単勝期待値・⭐の追加基準)", min_value=0.0, max_value=3.0, value=1.0, step=0.1,
            help="足切りされた馬(複勝確率が最低ライン未満)の中で、単勝期待値がこの数値を超えた馬は全員⭐にします。",
        )

    section_head("2", "出走馬の情報")

    if "horse_df" not in st.session_state:
        st.session_state.horse_df = DEFAULT_ROWS.copy()
    if "horse_table_version" not in st.session_state:
        st.session_state.horse_table_version = 0
    if "raw_paste_version" not in st.session_state:
        st.session_state.raw_paste_version = 0
    if "csv_paste_version" not in st.session_state:
        st.session_state.csv_paste_version = 0

    with st.expander("💡 出走馬をまとめて入力する(手入力の手間を減らせます)", expanded=True):
        tab1, tab2 = st.tabs(["netkeibaの出馬表を貼り付け", "CSVを貼り付け"])

        with tab1:
            st.caption(
                "netkeibaアプリ/サイトの「出馬表」ページで、表の部分を選択してコピーし、"
                "そのままここに貼り付けてください。"
            )
            raw_pasted = st.text_area(
                "netkeiba出馬表", height=150,
                key=f"raw_paste_{st.session_state.raw_paste_version}",
            )
            if st.button("表に反映", key="raw_apply"):
                apply_succeeded = False
                try:
                    st.session_state.horse_df = parse_netkeiba_shutuba(raw_pasted)
                    st.session_state.horse_table_version += 1
                    st.toast(f"{len(st.session_state.horse_df)}頭分を表に反映しました。", icon="✅")
                    st.session_state.raw_paste_version += 1
                    apply_succeeded = True
                except Exception as e:
                    st.error(str(e))
                if apply_succeeded:
                    st.rerun()

        with tab2:
            st.caption(
                "ヘッダー行付きのCSVを貼り付けてください。列は "
                "horse_num, waku, sex, age, jockey, weight_carry, horse_weight, "
                "weight_diff, prev_rank, rest_weeks, popularity, odds "
                "の一部だけでもOKです(無い列は既定値で埋めます)。"
            )
            pasted = st.text_area(
                "CSVテキスト", height=120,
                key=f"csv_paste_{st.session_state.csv_paste_version}",
            )
            if st.button("表に反映", key="csv_apply"):
                apply_succeeded = False
                try:
                    st.session_state.horse_df = parse_pasted_csv(pasted)
                    st.session_state.horse_table_version += 1
                    st.toast(f"{len(st.session_state.horse_df)}頭分を表に反映しました。", icon="✅")
                    st.session_state.csv_paste_version += 1
                    apply_succeeded = True
                except Exception as e:
                    st.error(f"読み込みに失敗しました: {e}")
                if apply_succeeded:
                    st.rerun()

    st.caption("表を直接編集できます。行の追加・削除も可能です。")

    edited = st.data_editor(
        st.session_state.horse_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "horse_num": st.column_config.NumberColumn("馬番", min_value=1, step=1),
            "horse_name": st.column_config.TextColumn("馬名"),
            "waku": st.column_config.NumberColumn("枠番", min_value=1, max_value=8, step=1),
            "sex": st.column_config.SelectboxColumn("性別", options=["牡", "牝", "セ"]),
            "age": st.column_config.NumberColumn("馬齢", min_value=2, max_value=10, step=1),
            "jockey": st.column_config.TextColumn("騎手", help="騎手名を直接入力してください"),
            "trainer": st.column_config.TextColumn("厩舎", help="例: 美浦 手塚久"),
            "running_style": st.column_config.SelectboxColumn("脚質", options=RUNNING_STYLES),
            "weight_carry": st.column_config.NumberColumn("斤量", min_value=48.0, max_value=64.0, step=0.5),
            "horse_weight": st.column_config.NumberColumn("馬体重", min_value=350, max_value=600, step=1),
            "weight_diff": st.column_config.NumberColumn("体重増減", min_value=-30, max_value=30, step=1),
            "prev_rank": st.column_config.NumberColumn("前走着順", min_value=1, max_value=18, step=1),
            "rest_weeks": st.column_config.NumberColumn("前走からの間隔(週)", min_value=1, max_value=52, step=1),
            "popularity": st.column_config.NumberColumn("人気", min_value=1, max_value=18, step=1),
            "odds": st.column_config.NumberColumn("オッズ", min_value=1.0, step=0.1, format="%.1f"),
        },
        key=f"horse_table_{st.session_state.horse_table_version}",
    )

    if st.button("🔎 馬名から前走成績を自動入力"):
        history, full_history = lookup_horse_history()
        if history.empty:
            st.session_state.autofill_msg = ("warning", "過去データが見つかりませんでした(まだ馬名付きのデータが少ない可能性があります)。")
            st.session_state.autofill_notes = {}
            st.session_state.autofill_suggestions = {}
        else:
            updated, matched, extra, suggestions = apply_horse_history(
                edited, history, full_history, turn_direction, hill, distance
            )
            st.session_state.horse_df = updated
            st.session_state.horse_table_version += 1
            st.session_state.prev_extra = extra
            notes = {}
            for n, e in extra.items():
                parts = []
                if e.get("prev_pace_note", PACE_NOTE_NONE) != PACE_NOTE_NONE:
                    parts.append(f"前走展開: {e['prev_pace_note']}")
                if e.get("prev_stretch_out_note", STRETCH_OUT_NOTE_NONE) != STRETCH_OUT_NOTE_NONE:
                    parts.append(f"注目: {e['prev_stretch_out_note']}")
                if e.get("horse_turn_aptitude", APTITUDE_UNKNOWN) not in (APTITUDE_UNKNOWN, "差なし"):
                    parts.append(f"回り適性: {e['horse_turn_aptitude']}")
                if e.get("horse_hill_aptitude", APTITUDE_UNKNOWN) not in (APTITUDE_UNKNOWN, "差なし"):
                    parts.append(f"坂適性: {e['horse_hill_aptitude']}")
                if e.get("horse_distance_aptitude", APTITUDE_UNKNOWN) not in (APTITUDE_UNKNOWN, "差なし"):
                    parts.append(f"距離適性: {e['horse_distance_aptitude']}")
                if parts:
                    notes[n] = " / ".join(parts)
            st.session_state.autofill_notes = notes
            st.session_state.autofill_suggestions = suggestions
            if matched:
                st.session_state.autofill_msg = ("success", f"{matched}頭分、前走成績を反映しました(表の「前走着順」を確認してください)。")
            else:
                st.session_state.autofill_msg = ("info", "表の馬名と一致する過去データが見つかりませんでした。馬名の表記が過去データと合っているか確認してください。")
        st.rerun()

    # ボタンを押した結果は、rerun後もこのブロックで表示し続ける(reranしても消えないように)
    if "autofill_msg" in st.session_state:
        kind, msg = st.session_state.autofill_msg
        getattr(st, kind)(msg)
        for name, note in st.session_state.get("autofill_notes", {}).items():
            st.caption(f"📝 {name}: 前走の展開評価 → {note}")
        for typed, close in st.session_state.get("autofill_suggestions", {}).items():
            st.warning(f"「{typed}」は見つかりませんでしたが、データ内に似た名前「{close}」があります。表記が違う可能性があります。")

    if st.button("予測する", type="primary"):
        if edited.empty:
            st.warning("出走馬の情報を入力してください。")
            return

        # 次のレースに備えて、出馬表/CSVの貼り付け欄は空にしておく
        # (今回の予測結果はこのまま下に表示されるので、ここではrerunしない。
        #  次の画面表示から新しいキーの空欄になる)
        st.session_state.raw_paste_version += 1
        st.session_state.csv_paste_version += 1

        df = edited.copy()
        df["venue"] = venue
        df["distance"] = distance
        df["track_type"] = track_type
        df["condition"] = condition
        df["day_bias"] = day_bias
        df["straight_length"] = straight_length
        df["turn_direction"] = turn_direction
        df["hill"] = hill
        df["race_class"] = race_class
        df["class_level"] = class_level
        df["gate_sensitive"] = "枠影響大" if is_gate_sensitive_course(venue, distance, track_type) else "通常"

        # 出走メンバーの脚質構成から、このレースの想定ペースを判定する
        # (「脚質不問モード」の場合は、脚質の推定精度が低い前提で、ペース判定をニュートラルにする)
        if ignore_pace:
            predicted_pace = RACE_PACE_MIDDLE
            df["race_pace"] = RACE_PACE_MIDDLE
            df["pace_fit"] = "五分"
        else:
            predicted_pace = estimate_race_pace(df["running_style"]) if "running_style" in df.columns else RACE_PACE_MIDDLE
            df["race_pace"] = predicted_pace
            df["pace_fit"] = df["running_style"].apply(lambda s: pace_style_fit(predicted_pace, s)) if "running_style" in df.columns else "五分"

        # 「馬名から前走成績を自動入力」で取得した裏データ(タイム・上がり3F・通過順・展開評価)をマージ
        prev_extra = st.session_state.get("prev_extra", {})
        for col in ("prev_time_sec", "prev_agari_3f", "prev_corner_pos", "prev_class_level", "prev_race_time_score"):
            df[col] = 0.0
        df["prev_pace_note"] = PACE_NOTE_NONE
        df["prev_field_strength_note"] = FIELD_NOTE_NONE
        df["prev_stretch_out_note"] = STRETCH_OUT_NOTE_NONE
        df["horse_turn_aptitude"] = APTITUDE_UNKNOWN
        df["horse_hill_aptitude"] = APTITUDE_UNKNOWN
        df["horse_distance_aptitude"] = APTITUDE_UNKNOWN
        if prev_extra and "horse_name" in df.columns:
            for i, row in df.iterrows():
                name = str(row.get("horse_name", "")).strip()
                if name in prev_extra:
                    for col, val in prev_extra[name].items():
                        df.at[i, col] = val

        X, _ = build_features(df, encoders=bundle["encoders"])
        # 複勝率・勝率は1つのモデル(1着/2・3着/圏外の3クラス)から導く。
        # 複勝率 = P(1着) + P(2・3着) という足し算の構造になっているため、
        # 「勝率が複勝率を上回る」という矛盾は起きようがない。
        proba, win_proba = derive_probas(bundle["model"], X)

        # ただし、この時点の値は「馬1頭ごとに独立」に計算したものなので、
        # レース全体で見た時に「勝率の合計が100%(1着になる馬は必ず1頭)」
        # 「複勝率の合計が300%(3着以内に入る馬は必ず3頭)」になる保証が無い。
        # そこでレース単位で比率を保ったまま正規化する。
        # (「勝率×オッズ」を正しい期待値として扱うには、勝率の合計が100%に
        #  なっていることが重要なので、こちらを優先する)
        win_sum = win_proba.sum()
        if win_sum > 0:
            win_proba = win_proba / win_sum
        top3_sum = proba.sum()
        if top3_sum > 0:
            proba = proba / top3_sum * min(3, len(proba))

        display_cols = ["horse_num", "horse_name", "waku", "jockey", "popularity", "odds"]
        result = edited[[c for c in display_cols if c in edited.columns]].copy()
        if "running_style" in df.columns:
            result["脚質"] = df["running_style"].values
        result["複勝確率(%)"] = (proba * 100).round(1)
        result["勝率(%)"] = (win_proba * 100).round(1)
        # 単勝期待値(勝率×オッズ)。参考表示として残す。
        if "odds" in result.columns:
            result["単勝期待値"] = (win_proba * result["odds"].fillna(0)).round(2)
        else:
            result["単勝期待値"] = 0.0

        # 単勝期待値をそのまま総合スコアとして使う(参考表示・◎の決定に使う)
        result["総合スコア"] = result["単勝期待値"]

        # 複勝確率が最低ライン(min_proba_for_mark)を下回る馬は「足切り」対象。
        # 極端な人気薄は勝率の推定自体が不安定で、期待値の数字だけが
        # 偶然大きく出てしまうことがあるための対策。
        eligible_mask = (result["複勝確率(%)"] >= min_proba_for_mark).values
        eligible = result[eligible_mask].copy()
        ineligible = result[~eligible_mask].copy()

        marks_map = {}  # 元のindex -> 印

        if len(eligible):
            # ◎: 足切りを通過した馬の中で、「単勝期待値×複勝確率^2」が一番高い馬。
            # 複勝確率を2乗することで、複勝確率が高い馬(人気サイド含む)により重みを
            # 置くようにしている。単純な掛け算(1乗)だと、極端なオッズの人気薄が
            # 選ばれやすい傾向があったための調整。
            axis_score = eligible["総合スコア"] * (eligible["複勝確率(%)"] / 100) ** 2
            axis_idx = axis_score.idxmax()
            marks_map[axis_idx] = "◎"

            # ○▲△: ◎以外を複勝確率が高い順に並べ、上位2頭を○▲、
            # それ以降は「相手ライン」を超えている限り全員△にする(頭数は可変)
            others = eligible.drop(index=axis_idx).sort_values("複勝確率(%)", ascending=False)
            aite_marks = ["○", "▲"]
            for rank, (idx, row) in enumerate(others.iterrows()):
                if rank < len(aite_marks):
                    marks_map[idx] = aite_marks[rank]
                elif row["複勝確率(%)"] >= aite_line:
                    marks_map[idx] = "△"
                else:
                    break  # 複勝確率順なので、ラインを割ったらそれ以降も全て割る

        # ⭐: 足切りされた馬の中で、単勝期待値が「穴ライン」を超えた馬は全員(頭数は可変)
        for idx, row in ineligible.iterrows():
            if row["総合スコア"] >= ana_line:
                marks_map[idx] = "⭐"

        # 印がついた馬を上(◎○▲△⭐の順)、それ以外を総合スコア順に並べ直す
        mark_priority = {"◎": 0, "○": 1, "▲": 2, "△": 3, "⭐": 4}
        result["印"] = result.index.map(lambda i: marks_map.get(i, ""))
        result["_並び順"] = result["印"].map(lambda m: mark_priority.get(m, 99))
        result = result.sort_values(["_並び順", "総合スコア"], ascending=[True, False]).drop(columns="_並び順").reset_index(drop=True)
        result = result[["印"] + [c for c in result.columns if c != "印"]]

        # AIの評価順位と人気のズレから「妙味」を判定する
        ai_rank = pd.Series(range(1, len(result) + 1), index=result.index)
        if "popularity" in result.columns:
            gap = compute_value_gap(ai_rank, result["popularity"])
            result["人気とのズレ"] = gap
            result["評価"] = gap.map(gap_label)

        result.index = ai_rank
        result = result.rename(columns={
            "horse_num": "馬番", "horse_name": "馬名", "waku": "枠番", "jockey": "騎手",
            "popularity": "人気", "odds": "オッズ",
        })

        # 表示部分はボタンの外(session_state経由)で行う。
        # ここでボタンブロック内のままにすると、下の買い目セレクトボックスを
        # 操作するたびに画面が再実行されて予測結果ごと消えてしまうため。
        st.session_state.prediction_result = result
        st.session_state.prediction_pace = predicted_pace
        st.session_state.prediction_ignore_pace = ignore_pace

    if "prediction_result" in st.session_state:
        result = st.session_state.prediction_result
        section_head("3", "予測結果")

        if "prediction_pace" in st.session_state:
            if st.session_state.get("prediction_ignore_pace"):
                st.caption("🏇 脚質不問モードのため、ペース判定はニュートラル(五分)にしています。")
            else:
                pace = st.session_state.prediction_pace
                pace_hint = {
                    "スロー": "逃げ・先行が有利になりやすい想定です",
                    "ミドル": "特定の脚質に大きく偏らない想定です",
                    "ハイ": "差し・追い込みが届きやすい想定です",
                }.get(pace, "")
                st.caption(f"🏇 出走メンバーの脚質構成からの想定ペース: **{pace}**({pace_hint})")

        # 上位ピックアップを見やすく表示
        picks = result[result["印"] != ""]
        cols = st.columns(len(picks))
        for col, (_, row) in zip(cols, picks.iterrows()):
            name = row.get("馬名") or f"{row['馬番']}番"
            col.metric(f"{row['印']} {row['馬番']}番", f"{row['複勝確率(%)']}%", str(name))

        st.dataframe(result, use_container_width=True)
        st.bar_chart(result.set_index("馬番")["複勝確率(%)"])

        st.caption(
            "「人気とのズレ」がプラスの馬は、AIが市場(人気)より高く評価している= 妙味のある穴馬候補です。"
            "マイナスの馬は人気先行の可能性があります。"
        )

        st.markdown("---")
        section_head("🐦", "ツイート文を作る")
        race_label = st.text_input(
            "レース名(【】の中に入る文字。空欄でもOK)",
            value=st.session_state.get("tweet_race_label", ""),
            key="tweet_race_label_input",
        )
        if st.button("ツイート文を生成", key="gen_tweet_btn"):
            st.session_state.tweet_text = generate_tweet_text(result, race_label)
        if "tweet_text" in st.session_state:
            st.text_area(
                "生成されたツイート文(コピーしてXに貼ってください)",
                value=st.session_state.tweet_text, height=180, key="tweet_text_display",
            )
            st.caption(f"文字数(概算): {_x_weight(st.session_state.tweet_text)} / 280")

        top3_picks = result.iloc[:3] if len(result) >= 3 else result
        if len(top3_picks) >= 2:
            st.markdown("---")
            section_head("4", "買い目の期待値を計算する")
            st.caption(
                "印がついた馬の中から、軸・相手を自由に選んで組み合わせを作れます。"
                "実際のオッズを入力すると期待値(確率×オッズ)を計算します。単勝オッズはデータに"
                "ありますが、それ以外の馬券種はオッズが分からないため、購入前にオッズ表アプリなどで"
                "確認して入力してください。期待値が1.0を超えるほど、理論上「買う価値がある」目安になります。"
            )

            # 印がついた全馬(◎○▲△△)を選択肢にする
            marked = result[result["印"] != ""].reset_index(drop=True)
            option_labels = [f"{row['印']}{row['馬番']}番 {row.get('馬名', '')}".strip() for _, row in marked.iterrows()]
            win_p_by_label = {
                label: marked.iloc[i]["勝率(%)"] / 100 for i, label in enumerate(option_labels)
            }
            odds_by_label = {
                label: float(marked.iloc[i].get("オッズ", 0) or 0) for i, label in enumerate(option_labels)
            }
            # 印がついていない残り馬の勝率(ワイドの計算に必要)
            unmarked = result[result["印"] == ""]
            others_list = (unmarked["勝率(%)"] / 100).tolist() if len(unmarked) else [0.0]

            st.markdown("**単勝**")
            tansho_choice = st.selectbox("単勝で買う馬", option_labels, key="tansho_choice")
            bets_input = [{
                "種類": "単勝", "対象": tansho_choice,
                "確率": win_p_by_label[tansho_choice], "オッズ": odds_by_label[tansho_choice],
            }]

            if len(option_labels) >= 2:
                st.markdown("**馬連・ワイド**(2頭選択)")
                col_a, col_b = st.columns(2)
                with col_a:
                    umaren_pair = st.multiselect(
                        "組み合わせる2頭", option_labels, default=option_labels[:2],
                        max_selections=2, key="umaren_pair",
                    )
                with col_b:
                    umaren_odds = st.number_input("馬連オッズ", min_value=0.0, step=0.1, key="umaren_odds_v2")
                    wide_odds = st.number_input("ワイドオッズ", min_value=0.0, step=0.1, key="wide_odds_v2")
                if len(umaren_pair) == 2:
                    pa, pb = win_p_by_label[umaren_pair[0]], win_p_by_label[umaren_pair[1]]
                    label_pair = f"{umaren_pair[0]}-{umaren_pair[1]}"
                    bets_input.append({"種類": "馬連", "対象": label_pair, "確率": quinella_proba(pa, pb), "オッズ": umaren_odds})
                    bets_input.append({"種類": "ワイド", "対象": label_pair, "確率": wide_proba(pa, pb, others_list), "オッズ": wide_odds})
                else:
                    st.caption("馬連・ワイドは、ちょうど2頭選んでください。")

            if len(option_labels) >= 3:
                st.markdown("**三連複**(3頭選択)")
                sanrenpuku_trio = st.multiselect(
                    "組み合わせる3頭", option_labels, default=option_labels[:3],
                    max_selections=3, key="sanrenpuku_trio",
                )
                sanrenpuku_odds = st.number_input("三連複オッズ", min_value=0.0, step=0.1, key="sanrenpuku_odds_v2")
                if len(sanrenpuku_trio) == 3:
                    pa, pb, pc = (win_p_by_label[n] for n in sanrenpuku_trio)
                    label_trio = "-".join(sanrenpuku_trio)
                    bets_input.append({"種類": "三連複", "対象": label_trio, "確率": trio_proba(pa, pb, pc), "オッズ": sanrenpuku_odds})
                else:
                    st.caption("三連複は、ちょうど3頭選んでください。")

                st.markdown("**三連単**(着順を指定)")
                col_1, col_2, col_3 = st.columns(3)
                with col_1:
                    santan_1st = st.selectbox("1着", option_labels, key="santan_1st")
                with col_2:
                    santan_2nd = st.selectbox("2着", option_labels, index=min(1, len(option_labels) - 1), key="santan_2nd")
                with col_3:
                    santan_3rd = st.selectbox("3着", option_labels, index=min(2, len(option_labels) - 1), key="santan_3rd")
                santan_odds = st.number_input("三連単オッズ", min_value=0.0, step=0.1, key="santan_odds_v2")
                if len({santan_1st, santan_2nd, santan_3rd}) == 3:
                    p1, p2, p3 = win_p_by_label[santan_1st], win_p_by_label[santan_2nd], win_p_by_label[santan_3rd]
                    bets_input.append({
                        "種類": "三連単", "対象": f"{santan_1st}→{santan_2nd}→{santan_3rd}",
                        "確率": trifecta_proba(p1, p2, p3), "オッズ": santan_odds,
                    })
                else:
                    st.caption("三連単は、1着・2着・3着に異なる馬を選んでください。")

            ev_result = compute_expected_values(bets_input)
            ev_df = pd.DataFrame(ev_result)
            ev_df["確率(%)"] = (ev_df["確率"] * 100).round(2)
            ev_df = ev_df[["種類", "対象", "確率(%)", "オッズ", "期待値"]]
            st.dataframe(ev_df, use_container_width=True, hide_index=True)
            st.caption(
                "確率はAIの推定(ハーヴィルの公式による近似)なので、あくまで目安です。"
                "オッズを入力していない馬券種は期待値が空欄のままになります。"
            )

        st.markdown(
            '<div class="disclaimer">⚠️ 本ツールは娯楽・分析目的の予測ツールです。'
            '馬券の購入は自己責任で、ご利用は20歳以上の方に限ります。</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
