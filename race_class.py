# -*- coding: utf-8 -*-
"""
レース格(新馬・未勝利・1勝クラス・オープン・重賞など)の判定と、
「どれくらいレベルが高いか」を表す数値(class_level)への変換をまとめたモジュール。

同じ着順・同じ人気でも、G1で3着とオープン特別で3着では価値が違う。
これを予測材料に組み込むための土台になる。
"""
import re

# (検索パターン, 表示名, レベル値) の順。上から順にマッチを試すので、
# 重賞(G1/G2/G3/L)を先に判定してから、クラス名を判定する。
RACE_CLASS_PATTERNS = [
    (r"\(?G1\)?|\(?GI\)?(?![IV])", "G1", 7),
    (r"\(?G2\)?|\(?GII\)?(?!I)", "G2", 6),
    (r"\(?G3\)?|\(?GIII\)?", "G3", 5),
    (r"\(L\)", "L(リステッド)", 4.5),
    (r"オープン特別|オープン|OP\b", "オープン", 4),
    (r"3勝クラス", "3勝クラス", 3),
    (r"2勝クラス", "2勝クラス", 2),
    (r"1勝クラス", "1勝クラス", 1),
    (r"未勝利", "未勝利", 0),
    (r"新馬", "新馬", 0),
]

# 重賞(G1/G2/G3)かどうかの判定に使う
GRADED_LABELS = {"G1", "G2", "G3"}


def detect_race_class(text: str):
    """レースのヘッダー文字列から、レース格(表示名)と数値レベルを判定する。

    見つからない場合は ("未勝利", 0) を返す(2歳新馬〜未勝利戦が最も多いための
    無難な既定値。手動で直したい場合はCSVのrace_class/class_levelを編集すればよい)。
    """
    for pattern, label, level in RACE_CLASS_PATTERNS:
        if re.search(pattern, text):
            return label, level
    return "未勝利", 0


def is_graded(race_class: str) -> bool:
    """重賞(G1/G2/G3)かどうかを返す。"""
    return race_class in GRADED_LABELS


def describe_race_level(race_class: str, race_time_score) -> str:
    """レース格・タイム水準から、「レベルが高い/普通/低い」を一言で表現する。

    重賞・オープン・Lなど、格自体で高レベルと分かる場合はそれを使う。
    新馬・未勝利・条件戦などは、タイム水準(race_time_score)で判定する。
    """
    if race_class in ("G1", "G2", "G3"):
        return f"レベル高い({race_class})"
    if race_class in ("L(リステッド)", "オープン"):
        return f"レベルやや高い({race_class})"

    try:
        score = float(race_time_score)
    except (TypeError, ValueError):
        score = 0.0

    if score >= 1.0:
        return "レベル高め(平均より速いタイム)"
    if score <= -1.0:
        return "レベル低め(平均より遅いタイム)"
    return "普通"
