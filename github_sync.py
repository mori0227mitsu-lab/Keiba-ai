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
    """既存CSVに新しい行を追記してコミットする。戻り値: 追記後の総行数"""
    df, sha = fetch_csv(token, repo, branch, path)

    # 列構成を既存CSVに合わせる(新しい列は追加、足りない列は0で埋める)
    for col in new_rows.columns:
        if col not in df.columns:
            df[col] = 0
    for col in df.columns:
        if col not in new_rows.columns:
            new_rows[col] = 0
    new_rows = new_rows[df.columns]

    combined = pd.concat([df, new_rows], ignore_index=True)
    update_csv(token, repo, branch, path, combined, sha, message)
    return len(combined)
