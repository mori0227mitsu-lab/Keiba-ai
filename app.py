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
)
from data_collector import (
    parse_netkeiba_result, parse_netkeiba_results_multi, apply_corner_section_to_df,
    fetch_netkeiba_text, fetch_and_parse_netkeiba_result,
)
from github_sync import append_rows_to_csv, fetch_csv, find_existing_race_ids, get_next_race_id
from race_class import RACE_CLASS_PATTERNS, describe_race_level
from course_info import (
    COURSE_DISTANCES,
    COURSE_HILL,
    COURSE_STRAIGHT_LENGTH,
    COURSE_TURN,
    DAY_BIAS_OPTIONS,
    RUNNING_STYLES,
    VENUES,
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
        if row.get("pace_note") == "強い勝ち方(先行して上がり負けでも勝利)":
            note_texts.append(
                "先行して上がりは平凡だったにも関わらず好走しています。展開関係なく地力で押し切れる、"
                "素質を感じさせる内容です。"
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

        # この馬の全過去成績から、今回のコース条件・距離との得意不得意を判定
        turn_apt, hill_apt, dist_apt = APTITUDE_UNKNOWN, APTITUDE_UNKNOWN, APTITUDE_UNKNOWN
        if not full_history.empty and "horse_name" in full_history.columns:
            name_hist = full_history[full_history["horse_name"] == actual_key]
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
            background: linear-gradient(135deg, #2F6F4E 0%, #1F4D37 100%);
            color: #FAF8F3;
            margin-bottom: 1.6rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 1.9rem;
            letter-spacing: 0.02em;
        }
        .hero p {
            margin: 0.3rem 0 0 0;
            color: #E4DCC8;
            font-size: 0.95rem;
        }
        .section-head {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-left: 5px solid #C9A227;
            padding-left: 0.6rem;
            margin: 1.4rem 0 0.6rem 0;
        }
        .section-head .num {
            background: #2F6F4E;
            color: #FAF8F3;
            border-radius: 50%;
            width: 1.6rem;
            height: 1.6rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .section-head .label {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1F2A24;
        }
        .disclaimer {
            font-size: 0.78rem;
            color: #8A8578;
            border-top: 1px solid #E7E2D6;
            padding-top: 0.6rem;
            margin-top: 1rem;
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
                "1回につき1レース分です。うまく取れない場合は上のコピペ方式を使ってください。"
            )
            result_url = st.text_input("結果ページのURL", key="result_url_input")
            if st.button("URLから取得してプレビュー", key="result_url_fetch"):
                try:
                    with st.spinner("取得中..."):
                        if use_auto_id:
                            token = st.secrets["github_token"]
                            repo = st.secrets["github_repo"]
                            branch = st.secrets.get("github_branch", "main")
                            existing_df_u, _ = fetch_csv(token, repo, branch, "data/dummy_races.csv")
                            start_id_u = (int(existing_df_u["race_id"].max()) + 1) if len(existing_df_u) else 9001
                        else:
                            start_id_u = int(manual_start_id)
                        url_preview_df = fetch_and_parse_netkeiba_result(
                            result_url, race_id=start_id_u, race_date=str(collect_date),
                        )
                    st.session_state.collect_preview = url_preview_df
                    st.toast(f"{len(url_preview_df)}頭分を取得しました。下のプレビューを確認してください。", icon="✅")
                    st.rerun()
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
            with st.expander("🔗 URLを貼るだけで取得する(実験的機能)"):
                st.caption(
                    "コピペの代わりに、netkeibaの出馬表ページのURLを貼って自動取得を試せます。"
                    "サイトの構造次第でうまく取れないこともあるので、その場合は上のコピペ方式を使ってください。"
                )
                shutuba_url = st.text_input("出馬表のURL", key="shutuba_url_input")
                if st.button("URLから取得", key="shutuba_url_fetch"):
                    try:
                        with st.spinner("取得中..."):
                            fetched_text = fetch_netkeiba_text(shutuba_url)
                        st.session_state.raw_paste_version += 1
                        st.session_state[f"raw_paste_{st.session_state.raw_paste_version}"] = fetched_text
                        st.toast("取得しました。下の欄に反映しています。", icon="✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"取得に失敗しました: {e}")
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
        proba = bundle["model"].predict_proba(X)[:, 1]

        # 勝率(1着になる確率)も計算し、レース全体で合計1になるよう正規化する
        # (単勝オッズが無い馬券種の期待値計算に使う)
        win_proba_raw = bundle["win_model"].predict_proba(X)[:, 1]
        win_proba_sum = win_proba_raw.sum()
        win_proba = win_proba_raw / win_proba_sum if win_proba_sum > 0 else win_proba_raw

        display_cols = ["horse_num", "horse_name", "waku", "jockey", "popularity", "odds"]
        result = edited[[c for c in display_cols if c in edited.columns]].copy()
        result["複勝確率(%)"] = (proba * 100).round(1)
        result["勝率(%)"] = (win_proba * 100).round(1)

        # AIの評価が高い順に並べ、上位に印をつける
        result = result.sort_values("複勝確率(%)", ascending=False).reset_index(drop=True)
        result.insert(0, "印", assign_marks(len(result)))

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

    if "prediction_result" in st.session_state:
        result = st.session_state.prediction_result
        section_head("3", "予測結果")

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
