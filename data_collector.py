# -*- coding: utf-8 -*-
"""
netkeibaの「結果」ページのコピペをCSV行(DataFrame)に変換するモジュール。

app.py の「データを集める」機能から呼び出す。
"""
import re

import pandas as pd

from course_info import COURSE_STRAIGHT_LENGTH

HEADER_RE = re.compile(
    r"発走\s*/\s*(?P<track>芝|ダ)(?P<distance>\d+)m.*?/\s*馬場:(?P<condition>\S+)\n"
    r"(?:.*\n)?"
    r"\d+回\s*(?P<venue>\S+?)\s*\d+日目",
)

HORSE_RE = re.compile(
    r"^(?P<finish>\d+|中止|除外|取消)\n"
    r"(?P<waku>\d+)\n"
    r"(?P<num>\d+)\n"
    r"\s*(?P<name>[^\t\n]+?)\t?\n"
    r"(?P<sex>[牡牝セ])(?P<age>\d+)\n"
    r"(?P<wc>[\d.]+)\n"
    r"(?P<jockey>[^\t\n]+)\t(?P<time>[\d:.]*)\t[^\n]*\n"
    r"(?P<pop>\d+)\n"
    r"(?P<odds>[\d.]*)\t(?P<agari>[\d.]*)\t(?P<corner>[\d\-]*)\t(?P<trainer>[^\t\n]+)\t(?P<weight>\d+)\((?P<wdiff>[+-]?\d+)\)",
    re.MULTILINE,
)

# 「コーナー通過順位」セクション(表とは別に、レースページ下部に載っている形式)
# 例:
#   コーナー通過順位
#   3コーナー	(*1,3)(7,8)(2,6)5=(4,9)
#   4コーナー	(*1,3)(6,7)8,2-5-4-9
CORNER_LINE_RE = re.compile(r"([1-4])コーナー\t([^\n]*)\n")

MARKS = "▲△☆★◇"
COND_FIX = {"稍": "稍重", "不": "不良"}

CSV_COLS = [
    "race_id", "race_date", "venue", "distance", "track_type", "condition", "straight_length",
    "day_bias", "horse_num", "horse_name", "waku", "sex", "age", "jockey", "trainer",
    "running_style", "weight_carry", "horse_weight", "weight_diff",
    "prev_rank", "rest_weeks", "popularity", "odds", "finish_rank",
    "time_sec", "agari_3f", "corner_pos",
]


def _clean(s: str) -> str:
    s = s.strip()
    for m in MARKS:
        s = s.replace(m, "")
    return s.strip()


def _time_to_sec(t: str):
    if not t:
        return None
    m = re.match(r"(?:(\d+):)?([\d.]+)$", t)
    if not m:
        return None
    mins = int(m.group(1)) if m.group(1) else 0
    return round(mins * 60 + float(m.group(2)), 1)


def _styles_from_corner(pos_list):
    """コーナー通過順位から、レース内での相対的な脚質を判定する。

    全馬とも通過順位が無い(netkeiba側にまだ反映されていない等)場合は、
    でたらめな判定を避けるため、全馬Noneを返す(呼び出し側で「不明」扱いにする)。
    """
    n = len(pos_list)
    if all(p == 999 for p in pos_list):
        return [None] * n
    order = sorted(range(n), key=lambda i: pos_list[i])
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r + 1
    out = []
    for r in rank:
        p = r / n
        out.append("逃げ" if p <= 0.25 else "先行" if p <= 0.5 else "差し" if p <= 0.75 else "追い込み")
    return out


def _parse_corner_group_string(s: str) -> dict:
    """(*1,3)(7,8)(2,6)5=(4,9) のような表記から、馬番->順位(1が先頭)のdictを作る。

    カッコでまとめられた馬は「ほぼ同じ位置」を表すので、同じ順位番号にする。
    """
    s = s.replace("*", "")
    tokens = re.findall(r"\([\d,]+\)|\d+", s)
    result = {}
    rank = 0
    for tok in tokens:
        rank += 1
        nums = [int(x) for x in re.findall(r"\d+", tok)]
        for n in nums:
            result[n] = rank
    return result


def _extract_corner_section(block: str) -> dict:
    """ブロック内に「コーナー通過順位」セクションがあれば、
    最終コーナー(一番大きい番号のもの)の内容を 馬番->順位 のdictにして返す。
    セクションが無い/内容が空の場合は空dictを返す。
    """
    lines = dict(CORNER_LINE_RE.findall(block))
    for corner_num in ("4", "3", "2", "1"):
        content = lines.get(corner_num, "").strip()
        if content:
            return _parse_corner_group_string(content)
    return {}


def apply_corner_section_to_df(df: pd.DataFrame, corner_text: str, race_id: int) -> pd.DataFrame:
    """既に作成済みのDataFrame(プレビュー結果)に対して、
    あとから貼り付けた「コーナー通過順位」セクションを反映する。

    指定したrace_idの行だけを対象に、馬番から通過順位を引いてcorner_posを更新し、
    そのレース内のrunning_styleも再計算する。
    """
    df = df.copy()
    corner_section = _extract_corner_section(corner_text)
    if not corner_section:
        raise ValueError(
            "「Nコーナー\\t(内容)」の形式を認識できませんでした。"
            "「コーナー通過順位」の行から、1〜4コーナー分をまとめて貼り付けてください。"
        )

    mask = df["race_id"] == race_id
    if not mask.any():
        raise ValueError(f"race_id={race_id} の行が見つかりませんでした。")

    sub = df.loc[mask].copy()
    updated_count = 0
    for i in sub.index:
        num = int(sub.at[i, "horse_num"])
        if num in corner_section:
            df.at[i, "corner_pos"] = corner_section[num]
            updated_count += 1

    if updated_count == 0:
        raise ValueError("貼り付けたコーナー通過順位に、該当する馬番が見つかりませんでした。")

    # このレース内で脚質を再計算する
    sub = df.loc[mask]
    corners = [int(c) if pd.notna(c) else 999 for c in sub["corner_pos"]]
    styles = _styles_from_corner(corners)
    for i, style in zip(sub.index, styles):
        df.at[i, "running_style"] = style if style is not None else "不明"

    return df


def _parse_one_block(block: str, race_id: int, race_date: str = "") -> pd.DataFrame:
    """1レース分のブロックをCSV_COLS形式のDataFrameに変換する(内部用)。"""
    hm = HEADER_RE.search(block)
    if not hm:
        raise ValueError(
            "レース条件(距離・コース種別・馬場状態・開催場)を認識できませんでした。"
            "ページ上部の「発走」を含む行が含まれているか確認してください。"
        )
    venue = hm.group("venue")
    condition = COND_FIX.get(hm.group("condition"), hm.group("condition"))
    track_type = "芝" if hm.group("track") == "芝" else "ダート"
    distance = hm.group("distance")
    straight = COURSE_STRAIGHT_LENGTH.get(venue, "普通")

    horses = list(HORSE_RE.finditer(block))
    if not horses:
        raise ValueError(
            "出走馬の結果テーブルを認識できませんでした。ページのレイアウトが"
            "想定と違う可能性があります。"
        )

    corner_section = _extract_corner_section(block)

    corners = []
    for h in horses:
        c = h.group("corner")
        last = c.split("-")[-1] if c else ""
        if last.isdigit():
            corners.append(int(last))
        elif corner_section and int(h.group("num")) in corner_section:
            # 行内に無ければ、「コーナー通過順位」セクションの値で補う
            corners.append(corner_section[int(h.group("num"))])
        else:
            corners.append(999)
    styles = _styles_from_corner(corners)

    rows = []
    for h, style, cpos in zip(horses, styles, corners):
        finish = h.group("finish")
        finish = "99" if finish in ("中止", "除外", "取消") else finish
        rows.append({
            "race_id": race_id,
            "race_date": race_date,
            "venue": venue,
            "distance": int(distance),
            "track_type": track_type,
            "condition": condition,
            "straight_length": straight,
            "day_bias": "フラット",
            "horse_num": int(h.group("num")),
            "horse_name": _clean(h.group("name")),
            "waku": int(h.group("waku")),
            "sex": h.group("sex"),
            "age": int(h.group("age")),
            "jockey": _clean(h.group("jockey")),
            "trainer": _clean(h.group("trainer")),
            "running_style": style if style is not None else "不明",
            "weight_carry": float(h.group("wc")),
            "horse_weight": int(h.group("weight")),
            "weight_diff": int(h.group("wdiff")),
            "prev_rank": 0,
            "rest_weeks": 0,
            "popularity": int(h.group("pop")),
            "odds": float(h.group("odds")) if h.group("odds") else 0.0,
            "finish_rank": int(finish),
            "time_sec": _time_to_sec(h.group("time")),
            "agari_3f": float(h.group("agari")) if h.group("agari") else None,
            "corner_pos": cpos if cpos != 999 else None,
        })

    return pd.DataFrame(rows)[CSV_COLS]


def parse_netkeiba_result(text: str, race_id: int, race_date: str = "") -> pd.DataFrame:
    """netkeibaの結果ページの生テキストを、CSV_COLS形式のDataFrameに変換する。

    1レース分のテキストを想定(複数レースが混ざっている場合は最初の1レース分のみ)。
    複数レースまとめて処理したい場合は parse_netkeiba_results_multi() を使う。
    """
    return _parse_one_block(text, race_id, race_date)


def parse_netkeiba_results_multi(text: str, start_race_id: int, race_date: str = "") -> pd.DataFrame:
    """複数レース分のnetkeiba結果ページを、まとめて貼り付けても一括処理する。

    「発走 / 芝1200m ... 馬場:良」のようなヘッダー行が出現する回数だけ
    レースがあるとみなし、race_idは start_race_id から1つずつ増やして割り振る。
    race_date(例: "2026-07-25")を渡すと、前走の並び順の判定に使われる
    (race_idの割り振り順ではなく、実際の日付を基準に「前走」を特定できるようになる)。
    """
    header_matches = list(HEADER_RE.finditer(text))
    if not header_matches:
        raise ValueError(
            "レース条件(「発走 / 芝1200m ... 馬場:良」のような行)を1つも認識できませんでした。"
        )

    all_rows = []
    for i, hm in enumerate(header_matches):
        start = hm.start()
        end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
        block = text[start:end]
        race_id = start_race_id + i
        try:
            race_df = _parse_one_block(block, race_id, race_date)
            all_rows.append(race_df)
        except ValueError as e:
            raise ValueError(f"{i + 1}番目のレース({race_id}番)の解析に失敗しました: {e}")

    return pd.concat(all_rows, ignore_index=True)
