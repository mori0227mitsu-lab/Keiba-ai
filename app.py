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

from model.train_model import FEATURE_COLS, RAW_REQUIRED_COLS, build_features, compute_pace_note, PACE_NOTE_NONE
from data_collector import parse_netkeiba_result
from github_sync import append_rows_to_csv
from course_info import (
    COURSE_DISTANCES,
    COURSE_STRAIGHT_LENGTH,
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


def lookup_horse_history() -> pd.DataFrame:
    """data/dummy_races.csv(実データ)から、各馬名の最新レース結果を取得する。

    horse_nameをキーにした DataFrame を返す(prev_rank, time_sec, agari_3f, corner_pos)。
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

    hist = compute_pace_note(hist)  # 各レース内での展開評価(強い勝ち方/展開不利)を付与
    hist = hist.sort_values("race_id")
    latest = hist.groupby("horse_name", as_index=True).tail(1).set_index("horse_name")
    return latest


def _normalize_name(name: str) -> str:
    """馬名の表記ゆれ(全角/半角スペースなど)を吸収するための正規化。"""
    return str(name).strip().replace("\u3000", "").replace(" ", "")


def apply_horse_history(df: pd.DataFrame, history: pd.DataFrame):
    """出走馬テーブルのhorse_nameを使って前走成績(prev_rank)を自動入力する。

    見つかった前走のタイム・上がり3F・通過順は、表には出さず
    st.session_state.prev_extra に保存しておき、予測時にこっそり使う。
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
        extra[raw_name] = {
            "prev_time_sec": h.get("time_sec", 0) if pd.notna(h.get("time_sec")) else 0,
            "prev_agari_3f": h.get("agari_3f", 0) if pd.notna(h.get("agari_3f")) else 0,
            "prev_corner_pos": h.get("corner_pos", 0) if pd.notna(h.get("corner_pos")) else 0,
            "prev_pace_note": h.get("pace_note", PACE_NOTE_NONE) if pd.notna(h.get("pace_note")) else PACE_NOTE_NONE,
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

    with st.expander("📥 データを集める(結果ページを貼り付けて自動保存)"):
        st.caption(
            "netkeibaの「結果」ページの表をコピーして貼り付けると、解析してGitHub上の"
            "data/dummy_races.csvに追記できます(Streamlit Cloud側にgithub_tokenの設定が必要です)。"
        )
        collect_race_id = st.number_input("この結果のrace_id(既存と重複しない番号)", min_value=1, value=9001, step=1)
        collect_text = st.text_area("netkeibaの結果ページ", height=150, key="collect_paste")

        if st.button("プレビュー"):
            try:
                preview_df = parse_netkeiba_result(collect_text, race_id=int(collect_race_id))
                st.session_state.collect_preview = preview_df
                st.success(f"{len(preview_df)}頭分を解析しました。内容を確認してから保存してください。")
            except Exception as e:
                st.error(str(e))

        if "collect_preview" in st.session_state:
            st.dataframe(st.session_state.collect_preview, use_container_width=True)
            if st.button("✅ GitHubに保存する", type="primary"):
                try:
                    token = st.secrets["github_token"]
                    repo = st.secrets["github_repo"]
                    branch = st.secrets.get("github_branch", "main")
                    total = append_rows_to_csv(
                        token, repo, branch, "data/dummy_races.csv",
                        st.session_state.collect_preview,
                        message=f"Add race {int(collect_race_id)} data",
                    )
                    st.success(f"GitHubに保存しました!(CSV全体: {total}行)数分後にアプリが再デプロイされます。")
                    del st.session_state.collect_preview
                except KeyError:
                    st.error(
                        "GitHubのトークンが設定されていません。Streamlit Cloudの「Manage app」→"
                        "「Settings」→「Secrets」に github_token / github_repo を設定してください。"
                    )
                except Exception as e:
                    st.error(f"保存に失敗しました: {e}")

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
    day_bias = st.selectbox(
        "今日の馬場傾向",
        DAY_BIAS_OPTIONS,
        help="レース当日の実況・データを見て、内外や脚質の有利不利があれば選んでください。分からなければ「フラット」でOKです。",
    )
    st.caption(f"📏 {venue}の直線の長さ: {straight_length}(固定情報として自動反映されます)")

    section_head("2", "出走馬の情報")

    if "horse_df" not in st.session_state:
        st.session_state.horse_df = DEFAULT_ROWS.copy()
    if "horse_table_version" not in st.session_state:
        st.session_state.horse_table_version = 0

    with st.expander("💡 出走馬をまとめて入力する(手入力の手間を減らせます)", expanded=True):
        tab1, tab2 = st.tabs(["netkeibaの出馬表を貼り付け", "CSVを貼り付け"])

        with tab1:
            st.caption(
                "netkeibaアプリ/サイトの「出馬表」ページで、表の部分を選択してコピーし、"
                "そのままここに貼り付けてください。"
            )
            raw_pasted = st.text_area("netkeiba出馬表", height=150, key="raw_paste")
            if st.button("表に反映", key="raw_apply"):
                try:
                    st.session_state.horse_df = parse_netkeiba_shutuba(raw_pasted)
                    st.session_state.horse_table_version += 1
                    st.success(f"{len(st.session_state.horse_df)}頭分を表に反映しました。")
                except Exception as e:
                    st.error(str(e))

        with tab2:
            st.caption(
                "ヘッダー行付きのCSVを貼り付けてください。列は "
                "horse_num, waku, sex, age, jockey, weight_carry, horse_weight, "
                "weight_diff, prev_rank, rest_weeks, popularity, odds "
                "の一部だけでもOKです(無い列は既定値で埋めます)。"
            )
            pasted = st.text_area("CSVテキスト", height=120, key="csv_paste")
            if st.button("表に反映", key="csv_apply"):
                try:
                    st.session_state.horse_df = parse_pasted_csv(pasted)
                    st.session_state.horse_table_version += 1
                    st.success(f"{len(st.session_state.horse_df)}頭分を表に反映しました。")
                except Exception as e:
                    st.error(f"読み込みに失敗しました: {e}")

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
        history = lookup_horse_history()
        if history.empty:
            st.warning("過去データが見つかりませんでした(まだ馬名付きのデータが少ない可能性があります)。")
        else:
            updated, matched, extra, suggestions = apply_horse_history(edited, history)
            st.session_state.horse_df = updated
            st.session_state.horse_table_version += 1
            st.session_state.prev_extra = extra
            if matched:
                st.success(f"{matched}頭分、前走成績を反映しました(表の「前走着順」を確認してください)。")
                notes = {n: e["prev_pace_note"] for n, e in extra.items() if e.get("prev_pace_note", PACE_NOTE_NONE) != PACE_NOTE_NONE}
                for name, note in notes.items():
                    st.caption(f"📝 {name}: 前走の展開評価 → {note}")
            else:
                st.info("表の馬名と一致する過去データが見つかりませんでした。馬名の表記が過去データと合っているか確認してください。")
            if suggestions:
                for typed, close in suggestions.items():
                    st.warning(f"「{typed}」は見つかりませんでしたが、データ内に似た名前「{close}」があります。表記が違う可能性があります。")

    if st.button("予測する", type="primary"):
        if edited.empty:
            st.warning("出走馬の情報を入力してください。")
            return

        df = edited.copy()
        df["venue"] = venue
        df["distance"] = distance
        df["track_type"] = track_type
        df["condition"] = condition
        df["day_bias"] = day_bias
        df["straight_length"] = straight_length

        # 「馬名から前走成績を自動入力」で取得した裏データ(タイム・上がり3F・通過順・展開評価)をマージ
        prev_extra = st.session_state.get("prev_extra", {})
        for col in ("prev_time_sec", "prev_agari_3f", "prev_corner_pos"):
            df[col] = 0.0
        df["prev_pace_note"] = PACE_NOTE_NONE
        if prev_extra and "horse_name" in df.columns:
            for i, row in df.iterrows():
                name = str(row.get("horse_name", "")).strip()
                if name in prev_extra:
                    for col, val in prev_extra[name].items():
                        df.at[i, col] = val

        X, _ = build_features(df, encoders=bundle["encoders"])
        proba = bundle["model"].predict_proba(X)[:, 1]

        display_cols = ["horse_num", "horse_name", "waku", "jockey", "popularity", "odds"]
        result = edited[[c for c in display_cols if c in edited.columns]].copy()
        result["複勝確率(%)"] = (proba * 100).round(1)

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

        st.markdown(
            '<div class="disclaimer">⚠️ 本ツールは娯楽・分析目的の予測ツールです。'
            '馬券の購入は自己責任で、ご利用は20歳以上の方に限ります。</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
