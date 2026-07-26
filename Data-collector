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

MARKS = "▲△☆★◇"
COND_FIX = {"稍": "稍重", "不": "不良"}

CSV_COLS = [
    "race_id", "venue", "distance", "track_type", "condition", "straight_length",
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
    n = len(pos_list)
    order = sorted(range(n), key=lambda i: pos_list[i])
    rank = [0] * n
    for r, i in enumerate(order):
        rank[i] = r + 1
    out = []
    for r in rank:
        p = r / n
        out.append("逃げ" if p <= 0.25 else "先行" if p <= 0.5 else "差し" if p <= 0.75 else "追い込み")
    return out


def parse_netkeiba_result(text: str, race_id: int) -> pd.DataFrame:
    """netkeibaの結果ページの生テキストを、CSV_COLS形式のDataFrameに変換する。

    1レース分のテキストを想定(複数レースが混ざっている場合は最初の1レース分のみ)。
    """
    hm = HEADER_RE.search(text)
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

    horses = list(HORSE_RE.finditer(text))
    if not horses:
        raise ValueError(
            "出走馬の結果テーブルを認識できませんでした。ページのレイアウトが"
            "想定と違う可能性があります。"
        )

    corners = []
    for h in horses:
        c = h.group("corner")
        last = c.split("-")[-1] if c else ""
        corners.append(int(last) if last.isdigit() else 999)
    styles = _styles_from_corner(corners)

    rows = []
    for h, style, cpos in zip(horses, styles, corners):
        finish = h.group("finish")
        finish = "99" if finish in ("中止", "除外", "取消") else finish
        rows.append({
            "race_id": race_id,
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
            "running_style": style,
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
