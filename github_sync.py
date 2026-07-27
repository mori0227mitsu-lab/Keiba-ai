# -*- coding: utf-8 -*-
"""
GitHub API経由でリポジトリ内のCSVファイルを読み書きするモジュール。

st.secrets に github_token / github_repo / github_branch が
設定されていることを前提にする(app.py側から呼び出す)。
"""
import base64
import io

import pandas as pd
import requests

API_BASE = "https://api.github.com"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }


def fetch_csv(token: str, repo: str, branch: str, path: str):
    """GitHub上のCSVファイルを取得する。戻り値: (DataFrame, sha)"""
    url = f"{API_BASE}/repos/{repo}/contents/{path}"
    resp = requests.get(url, headers=_headers(token), params={"ref": branch}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(content))
    return df, data["sha"]


def update_csv(token: str, repo: str, branch: str, path: str, df: pd.DataFrame, sha: str, message: str):
    """CSVファイルの中身を丸ごと置き換えてコミットする。"""
    csv_text = df.to_csv(index=False)
    content_b64 = base64.b64encode(csv_text.encode("utf-8-sig")).decode("ascii")
    url = f"{API_BASE}/repos/{repo}/contents/{path}"
    body = {
        "message": message,
        "content": content_b64,
        "sha": sha,
        "branch": branch,
    }
    resp = requests.put(url, headers=_headers(token), json=body, timeout=20)
    resp.raise_for_status()
    return resp.json()


def append_rows_to_csv(token: str, repo: str, branch: str, path: str, new_rows: pd.DataFrame, message: str) -> int:
    """既存CSVに新しい行を追記してコミットする。戻り値: 追記後の総行数

    new_rowsの中に、既存CSVと同じrace_idの行が含まれている場合は、
    その古い行を削除してから追記する(=上書きになる)。
    """
    df, sha = fetch_csv(token, repo, branch, path)

    # 列構成を既存CSVに合わせる(新しい列は追加、足りない列は0で埋める)
    # 列を揃える時の既定値(数値系の列は0、文字列系の列は空文字にする)
    text_like_cols = {"race_date", "horse_name", "jockey", "trainer", "venue", "track_type", "condition"}

    def _default_for(col):
        return "" if col in text_like_cols else 0

    for col in new_rows.columns:
        if col not in df.columns:
            df[col] = _default_for(col)
    for col in df.columns:
        if col not in new_rows.columns:
            new_rows[col] = _default_for(col)
    new_rows = new_rows[df.columns]

    # 上書き対象(同じrace_id)の古い行を先に取り除く
    if "race_id" in df.columns and "race_id" in new_rows.columns:
        overwrite_ids = set(new_rows["race_id"].unique())
        df = df[~df["race_id"].isin(overwrite_ids)]

    combined = pd.concat([df, new_rows], ignore_index=True)
    update_csv(token, repo, branch, path, combined, sha, message)
    return len(combined)


def find_existing_race_ids(
    df: pd.DataFrame, races: list[dict],
) -> dict:
    """races(各レースの venue/distance/track_type/race_date/horse_names の辞書リスト)
    について、既存CSV(df)の中に同じ組み合わせのレースが無いか調べる。

    戻り値: {連番インデックス: 既存のrace_id} のdict(一致したものだけ)。
    race_date(開催日)が両方に入っている場合はそれも一致条件に加える。
    開催日が不明(空欄)な場合は、venue/distance/track_typeだけでは
    「偶然同じ条件の別レース」を誤って同一視してしまう恐れがあるため、
    出走馬名が実際に一定数重なっているかも必ず確認する。
    """
    matches = {}
    if df.empty or "venue" not in df.columns:
        return matches

    has_name_col = "horse_name" in df.columns

    for i, race in enumerate(races):
        cond = (
            (df["venue"] == race.get("venue"))
            & (df["distance"] == race.get("distance"))
            & (df["track_type"] == race.get("track_type"))
        )
        race_date = race.get("race_date")
        candidate = df[cond]
        if candidate.empty:
            continue

        new_names = set(race.get("horse_names") or [])

        for existing_race_id in candidate["race_id"].unique():
            sub = candidate[candidate["race_id"] == existing_race_id]
            existing_date_raw = str(sub["race_date"].iloc[0]) if "race_date" in sub.columns else ""
            existing_date = existing_date_raw.strip()
            if existing_date in ("nan", "NaT", "None", "0", "0.0"):
                existing_date = ""

            if race_date and existing_date:
                # 両方に開催日があるなら、日付が一致する場合だけ同一レースとみなす
                if existing_date == str(race_date):
                    matches[i] = int(existing_race_id)
                    break
                continue

            # どちらかの開催日が不明な場合は、出走馬名の重なりで確認する
            # (venue/distance/track_typeだけの一致は、偶然の可能性があるため)
            if has_name_col and new_names:
                existing_names = set(sub["horse_name"].dropna().astype(str))
                overlap = new_names & existing_names
                # 半分以上の馬名が重なっていれば、同一レースとみなす
                if len(overlap) >= max(1, len(new_names) // 2):
                    matches[i] = int(existing_race_id)
                    break
    return matches


def get_next_race_id(token: str, repo: str, branch: str, path: str, default: int = 9001) -> int:
    """既存CSVの最大race_idの次の番号を返す(race_idの重複・手入力ミスを防ぐため)。"""
    try:
        df, _ = fetch_csv(token, repo, branch, path)
        if "race_id" in df.columns and len(df):
            return int(df["race_id"].max()) + 1
    except Exception:
        pass
    return default
