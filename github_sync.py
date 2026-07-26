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
    for col in new_rows.columns:
        if col not in df.columns:
            df[col] = 0
    for col in df.columns:
        if col not in new_rows.columns:
            new_rows[col] = 0
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
    """races(各レースの venue/distance/track_type/race_date の辞書リスト)について、
    既存CSV(df)の中に同じ組み合わせのレースが無いか調べる。

    戻り値: {連番インデックス: 既存のrace_id} のdict(一致したものだけ)。
    race_date(開催日)が両方に入っている場合はそれも一致条件に加える
    (無い場合は venue/distance/track_typeだけで判定する)。
    """
    matches = {}
    if df.empty or "venue" not in df.columns:
        return matches

    for i, race in enumerate(races):
        cond = (
            (df["venue"] == race.get("venue"))
            & (df["distance"] == race.get("distance"))
            & (df["track_type"] == race.get("track_type"))
        )
        race_date = race.get("race_date")
        if race_date and "race_date" in df.columns:
            existing_date = df["race_date"].astype(str).str.strip()
            existing_date = existing_date.replace({"nan": "", "NaT": "", "None": ""})
            # 既存データの開催日が「空欄(不明)」の行は、日付が違っていても一致とみなす
            # (前回、日付を入力せずに保存したレースを、後から日付ありで上書きできるように)
            date_ok = (existing_date == "") | (existing_date == str(race_date))
            cond = cond & date_ok
        found = df[cond]
        if len(found):
            matches[i] = int(found["race_id"].iloc[0])
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
