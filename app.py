# -*- coding: utf-8 -*-
"""
競馬予想AI - Streamlit Webアプリ

使い方:
    streamlit run app.py

出走馬の情報を表として入力すると、学習済みモデルが
各馬の「複勝(3着以内)確率」を予測してランキング表示します。
"""

import os

import joblib
import pandas as pd
import streamlit as st

from model.train_model import build_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.joblib")

JOCKEYS = [
    "C.ルメール", "川田将雅", "武豊", "戸崎圭太", "横山武史",
    "福永祐一", "坂井瑠星", "松山弘平", "池添謙一", "岩田望来",
]

DEFAULT_ROWS = pd.DataFrame([
    {"horse_num": 1, "waku": 1, "sex": "牡", "age": 4, "jockey": "C.ルメール",
     "weight_carry": 57.0, "horse_weight": 480, "weight_diff": 2,
     "prev_rank": 2, "rest_weeks": 5, "popularity": 1, "odds": 2.5},
    {"horse_num": 2, "waku": 1, "sex": "牝", "age": 3, "jockey": "川田将雅",
     "weight_carry": 54.0, "horse_weight": 452, "weight_diff": -4,
     "prev_rank": 5, "rest_weeks": 8, "popularity": 2, "odds": 5.1},
    {"horse_num": 3, "waku": 2, "sex": "牡", "age": 5, "jockey": "武豊",
     "weight_carry": 58.0, "horse_weight": 498, "weight_diff": 0,
     "prev_rank": 1, "rest_weeks": 4, "popularity": 3, "odds": 6.8},
])


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dummy_races.csv")


@st.cache_resource
def load_model():
    """model.joblib が無ければ、その場でダミーデータ生成→学習して作る。

    GitHub上にモデルファイル(バイナリ)をアップロードしなくても
    アプリが自力でセットアップできるようにするための仕組み。
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    from data.generate_dummy_data import generate
    from model.train_model import train

    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        generate(DATA_PATH, verbose=False)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    return train(DATA_PATH, MODEL_PATH, verbose=False)


def main():
    st.set_page_config(page_title="競馬予想AI", page_icon="🐎", layout="wide")
    st.title("🐎 競馬予想AI")
    st.caption(
        "現在は練習用のダミーデータで学習したモデルです。"
        "実データ(netkeiba / JRA-VANなど)に差し替えることで精度が上がります。"
    )

    with st.spinner("モデルを準備しています(初回のみ数秒かかります)..."):
        bundle = load_model()

    st.subheader("① レース条件")
    col1, col2, col3 = st.columns(3)
    with col1:
        distance = st.number_input("距離(m)", min_value=1000, max_value=3600, value=1600, step=100)
    with col2:
        track_type = st.selectbox("コース種別", ["芝", "ダート"])
    with col3:
        condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])

    st.subheader("② 出走馬の情報")
    st.caption("表を直接編集できます。行の追加・削除も可能です。")

    edited = st.data_editor(
        DEFAULT_ROWS,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "horse_num": st.column_config.NumberColumn("馬番", min_value=1, step=1),
            "waku": st.column_config.NumberColumn("枠番", min_value=1, max_value=8, step=1),
            "sex": st.column_config.SelectboxColumn("性別", options=["牡", "牝", "セ"]),
            "age": st.column_config.NumberColumn("馬齢", min_value=2, max_value=10, step=1),
            "jockey": st.column_config.SelectboxColumn("騎手", options=JOCKEYS),
            "weight_carry": st.column_config.NumberColumn("斤量", min_value=48.0, max_value=64.0, step=0.5),
            "horse_weight": st.column_config.NumberColumn("馬体重", min_value=350, max_value=600, step=1),
            "weight_diff": st.column_config.NumberColumn("体重増減", min_value=-30, max_value=30, step=1),
            "prev_rank": st.column_config.NumberColumn("前走着順", min_value=1, max_value=18, step=1),
            "rest_weeks": st.column_config.NumberColumn("前走からの間隔(週)", min_value=1, max_value=52, step=1),
            "popularity": st.column_config.NumberColumn("人気", min_value=1, max_value=18, step=1),
            "odds": st.column_config.NumberColumn("オッズ", min_value=1.0, step=0.1, format="%.1f"),
        },
        key="horse_table",
    )

    if st.button("予測する", type="primary"):
        if edited.empty:
            st.warning("出走馬の情報を入力してください。")
            return

        df = edited.copy()
        df["distance"] = distance
        df["track_type"] = track_type
        df["condition"] = condition

        X, _ = build_features(df, encoders=bundle["encoders"])
        proba = bundle["model"].predict_proba(X)[:, 1]

        result = edited[["horse_num", "waku", "jockey", "popularity", "odds"]].copy()
        result["複勝確率(3着以内)"] = (proba * 100).round(1)
        result = result.sort_values("複勝確率(3着以内)", ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        result = result.rename(columns={
            "horse_num": "馬番", "waku": "枠番", "jockey": "騎手",
            "popularity": "人気", "odds": "オッズ",
        })

        st.subheader("③ 予測結果")
        st.dataframe(
            result.style.background_gradient(
                subset=["複勝確率(3着以内)"], cmap="YlOrRd"
            ),
            use_container_width=True,
        )
        st.bar_chart(result.set_index("馬番")["複勝確率(3着以内)"])

        st.caption(
            "⚠️ このアプリはあくまで学習・娯楽目的の予測ツールです。"
            "馬券の購入は自己責任で、ご利用は20歳以上の方に限ります。"
        )


if __name__ == "__main__":
    main()
