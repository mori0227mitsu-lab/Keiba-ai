# -*- coding: utf-8 -*-
"""
netkeibaの「結果」ページのコピペをCSV行(DataFrame)に変換するモジュール。

app.py の「データを集める」機能から呼び出す。
"""
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from course_info import COURSE_STRAIGHT_LENGTH
from race_class import detect_race_class

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# 出馬表・結果テーブルらしさを判定するためのキーワード
_TABLE_HEADER_KEYWORDS = [
    "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差",
    "人気", "オッズ", "厩舎", "馬体重",
]


def _find_race_table(soup: BeautifulSoup):
    """ページ内の<table>のうち、出馬表・結果テーブルらしいものを1つ選ぶ。

    見出しキーワードが多く含まれる表ほど「それらしい」とみなす。
    ページ内には過去の優勝馬一覧など紛らわしい表も他にあるため、
    単純に最初の表を使うと誤爆するのでスコアリングして選ぶ。
    """
    tables = soup.find_all("table")
    best_table, best_score = None, 0
    for table in tables:
        text = table.get_text()
        score = sum(1 for kw in _TABLE_HEADER_KEYWORDS if kw in text)
        # 出走頭数分のリンク(馬詳細ページへのリンク)が多いほど、それらしい表とみなす
        score += min(len(table.find_all("a")), 20) / 4
        if score > best_score:
            best_table, best_score = table, score
    return best_table


def _table_to_tsv_lines(table) -> list:
    """<table>を、1行=1レコードのタブ区切りテキストに変換する。

    空セル(着差が「クビ」等で無い勝ち馬など)を詰めてしまうと、後続の列が
    ズレて誤読の原因になるため、空セルも位置を保ったまま残す。
    (行の中身が完全に空の場合だけ、その行自体を捨てる)
    """
    lines = []
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        cell_texts = [c.get_text(separator=" ", strip=True) for c in cells]
        if any(t != "" for t in cell_texts):
            lines.append("\t".join(cell_texts))
    return lines


def fetch_netkeiba_text(url: str) -> str:
    """netkeibaのレースページ(出馬表・結果どちらも)をURLから取得し、

    表の中身をタブ・改行区切りのテキストに変換して返す(確認・デバッグ用)。
    実際のパースには fetch_and_parse_netkeiba_result() の方を使う。
    """
    soup = _fetch_soup(url, kind="shutuba")
    full_text = soup.get_text(separator="\n", strip=True)
    header_lines = [
        ln for ln in full_text.splitlines()
        if ("発走" in ln and ("m" in ln or "M" in ln)) or ("回" in ln and "日目" in ln)
    ]

    race_table = _find_race_table(soup)
    if race_table is None:
        raise ValueError(
            "出走馬の表が見つかりませんでした。ページの構造が想定と違う可能性があります。"
        )
    table_lines = _table_to_tsv_lines(race_table)

    return "\n".join(header_lines + table_lines)


def normalize_netkeiba_url(url: str, kind: str = "result") -> str:
    """スマホの「リンクをコピー」機能などで作られる様々な形式のnetkeiba URLから、
    race_idだけを取り出し、確実に読み込める形式のURLに組み直す。

    kind="result" なら結果ページ、kind="shutuba" なら出馬表ページのURLにする。
    """
    m = re.search(r"race_id=(\d+)", url)
    if not m:
        # race_idが見つからない場合は、元のURLをそのまま使う(万one形式が違うだけの場合に備えて)
        return url
    race_id = m.group(1)
    page = "shutuba.html" if kind == "shutuba" else "result.html"
    return f"https://race.sp.netkeiba.com/race/{page}?race_id={race_id}"


def _fetch_soup(url: str, kind: str = "result") -> BeautifulSoup:
    url = normalize_netkeiba_url(url, kind=kind)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"ページの取得に失敗しました: {e}")

    resp.encoding = resp.apparent_encoding or "euc-jp"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup


# 結果テーブルの列見出しを判定するためのキーワード → CSV_COLSでの役割 の対応
_COLUMN_ROLE_KEYWORDS = [
    ("finish", ["着順", "着 順"]),
    ("waku", ["枠"]),
    ("horse_num", ["馬番", "馬 番"]),
    ("name", ["馬名"]),
    ("sex_age", ["性齢"]),
    ("weight_carry", ["斤量"]),
    ("jockey", ["騎手"]),
    ("time", ["タイム"]),
    ("margin", ["着差"]),
    ("popularity", ["人気", "人 気"]),
    ("odds", ["オッズ"]),
    ("agari", ["後3F", "後3Ｆ"]),
    ("corner", ["通過順", "コーナー"]),
    ("trainer", ["厩舎"]),
    ("weight", ["馬体重"]),
]


def _map_columns(header_cells: list) -> dict:
    """ヘッダー行のセル文字列から、各列が何の役割かを判定してindexの辞書を作る。"""
    role_to_idx = {}
    used_idx = set()
    for role, keywords in _COLUMN_ROLE_KEYWORDS:
        for i, cell in enumerate(header_cells):
            if i in used_idx:
                continue
            if any(kw in cell for kw in keywords):
                role_to_idx[role] = i
                used_idx.add(i)
                break
    return role_to_idx


def _extract_header_info(soup: BeautifulSoup) -> dict:
    """ページ全体のテキストから、開催場・距離・コース種別・馬場状態を抜き出す。

    race_class判定用に、レース条件が書かれている周辺だけの短いテキスト
    (class_context)も一緒に返す。ページ全体をレース格判定に使うと、
    関係ない場所にある文字列(例えば単独の「L」)に誤反応する恐れがあるため、
    範囲を絞っておく。
    """
    flat = soup.get_text(separator=" ", strip=True)
    flat = re.sub(r"\s+", " ", flat)

    info = {"venue": None, "distance": None, "track_type": None, "condition": None, "class_context": ""}

    m = re.search(r"(芝|ダート|ダ)\s*(\d{3,4})\s*m", flat)
    if m:
        info["track_type"] = "芝" if m.group(1) == "芝" else "ダート"
        info["distance"] = int(m.group(2))

    m = re.search(r"馬場\s*:\s*(良|稍重|稍|重|不良|不)", flat)
    if m:
        info["condition"] = COND_FIX.get(m.group(1), m.group(1))

    m2 = re.search(r"\d+回\s*([^\s\d回]{2,4})\s*\d+日目", flat)
    if m2:
        info["venue"] = m2.group(1)
        # 「n回 会場 n日目」の直後(サラ系〜新馬/オープン/(G3)等が続く部分)だけを
        # レース格判定に使う。ページ冒頭(レース名にグレードが付くことがある)も少し含める。
        info["class_context"] = flat[:200] + " " + flat[m2.end():m2.end() + 150]

    return info


def fetch_and_parse_netkeiba_result(url: str, race_id: int, race_date: str = "") -> pd.DataFrame:
    """netkeibaの「結果」ページのURLから直接取得し、CSV_COLS形式のDataFrameを作る。

    テキストのコピペを経由せず、取得したHTMLの表を直接読み取るので、
    列のズレなどが起きにくい(はず)。
    """
    soup = _fetch_soup(url)
    header_info = _extract_header_info(soup)
    if not all([header_info["venue"], header_info["distance"], header_info["track_type"]]):
        raise ValueError(
            "レース条件(開催場・距離・コース種別)を読み取れませんでした。"
            "ページの構造が想定と違う可能性があります。"
        )

    race_table = _find_race_table(soup)
    if race_table is None:
        raise ValueError("出走馬の表が見つかりませんでした。")

    trs = race_table.find_all("tr")
    if len(trs) < 2:
        raise ValueError("出走馬の表の行数が想定より少ないです。")

    header_cells = [c.get_text(strip=True) for c in trs[0].find_all(["td", "th"])]
    role_to_idx = _map_columns(header_cells)
    required_roles = ["horse_num", "name", "sex_age", "jockey"]
    if not all(r in role_to_idx for r in required_roles):
        raise ValueError("表の列構成を認識できませんでした(見出しの形式が想定と違う可能性があります)。")

    venue = header_info["venue"]
    condition = header_info["condition"] or "良"
    track_type = header_info["track_type"]
    distance = header_info["distance"]
    straight = COURSE_STRAIGHT_LENGTH.get(venue, "普通")
    race_class, class_level = detect_race_class(header_info.get("class_context", ""))

    rows_raw = []
    for tr in trs[1:]:
        cells = [c.get_text(separator=" ", strip=True) for c in tr.find_all(["td", "th"])]
        if not any(c != "" for c in cells):
            continue

        def get(role, default=""):
            idx = role_to_idx.get(role)
            return cells[idx] if idx is not None and idx < len(cells) else default

        name = get("name")
        if not name:
            continue

        sex_age = get("sex_age")
        sex_m = re.match(r"([牡牝セ])(\d+)", sex_age)
        sex = sex_m.group(1) if sex_m else "牡"
        age = int(sex_m.group(2)) if sex_m else 0

        weight_m = re.match(r"(\d{2,3})\(([+-]?\d+)\)", get("weight").replace(" ", ""))
        weight = int(weight_m.group(1)) if weight_m else 0
        weight_diff = int(weight_m.group(2)) if weight_m else 0

        corner_raw = get("corner")
        corner_last = corner_raw.split("-")[-1] if corner_raw else ""
        corner_pos = int(corner_last) if corner_last.isdigit() else None

        finish_raw = get("finish")
        finish = 99 if finish_raw in ("中止", "除外", "取消") else (int(finish_raw) if finish_raw.isdigit() else 99)

        rows_raw.append({
            "finish": finish,
            "waku": int(get("waku") or 0),
            "horse_num": int(get("horse_num") or 0),
            "name": name,
            "sex": sex,
            "age": age,
            "weight_carry": float(get("weight_carry") or 0),
            "jockey": _clean(get("jockey")),
            "time": get("time"),
            "popularity": int(get("popularity") or 0),
            "odds": float(get("odds") or 0) if get("odds") else 0.0,
            "agari": float(get("agari")) if get("agari") else None,
            "corner": corner_pos,
            "trainer": _clean(get("trainer")),
            "weight": weight,
            "weight_diff": weight_diff,
        })

    if not rows_raw:
        raise ValueError("出走馬の行を1件も読み取れませんでした。")

    corners = [r["corner"] if r["corner"] is not None else 999 for r in rows_raw]
    styles = _styles_from_corner(corners)

    out_rows = []
    for r, style, cpos in zip(rows_raw, styles, corners):
        out_rows.append({
            "race_id": race_id,
            "race_date": race_date,
            "venue": venue,
            "distance": distance,
            "track_type": track_type,
            "condition": condition,
            "straight_length": straight,
            "day_bias": "フラット",
            "race_class": race_class,
            "class_level": class_level,
            "horse_num": r["horse_num"],
            "horse_name": r["name"],
            "waku": r["waku"],
            "sex": r["sex"],
            "age": r["age"],
            "jockey": r["jockey"],
            "trainer": r["trainer"],
            "running_style": style if style is not None else "不明",
            "weight_carry": r["weight_carry"],
            "horse_weight": r["weight"],
            "weight_diff": r["weight_diff"],
            "prev_rank": 0,
            "rest_weeks": 0,
            "popularity": r["popularity"],
            "odds": r["odds"],
            "finish_rank": r["finish"],
            "time_sec": _time_to_sec(r["time"]),
            "agari_3f": r["agari"],
            "corner_pos": cpos if cpos != 999 else None,
        })

    return pd.DataFrame(out_rows)[CSV_COLS]

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
    "day_bias", "race_class", "class_level", "horse_num", "horse_name", "waku", "sex", "age",
    "jockey", "trainer", "running_style", "weight_carry", "horse_weight", "weight_diff",
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
    race_class, class_level = detect_race_class(block)

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
            "race_class": race_class,
            "class_level": class_level,
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
