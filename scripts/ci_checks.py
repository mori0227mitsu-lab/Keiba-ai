# -*- coding: utf-8 -*-
"""
GitHub Actions(CI)で毎回自動実行される健康診断スクリプト。

コードを実行できないユーザーの代わりに、コミットのたびに
「壊れていないか」を機械的にチェックする。何か問題があれば
終了コード1で失敗させ、GitHub上で赤い✕マークが付くようにする。

チェック内容:
1. 主要モジュールが構文エラーなくimportできるか
2. 今のデータ(data/dummy_races.csv)で実際にモデルが学習できるか
3. データの基本的なスキーマが崩れていないか
4. 出走馬名がほぼ完全一致する「重複レース」が紛れ込んでいないか
5. 同じrace_idに複数の別レースが混在していないか(race_id衝突)
"""
import os
import sys

# リポジトリのルートをimportパスに追加(scripts/から見て一つ上の階層)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_PATH = os.path.join(ROOT, "data", "dummy_races.csv")

errors = []


def check(label, fn):
    """1項目のチェックを実行し、結果を表示する。失敗してもここでは止めず、
    最後にまとめて終了コードを決める(全項目の結果を一度に見せるため)。
    """
    print(f"[チェック] {label} ...", end=" ")
    try:
        fn()
        print("OK")
    except Exception as e:
        print("NG")
        print(f"  詳細: {e}")
        errors.append(label)


def check_imports():
    import course_info  # noqa: F401
    import race_class  # noqa: F401
    import data_collector  # noqa: F401
    import github_sync  # noqa: F401
    from model import train_model  # noqa: F401


def check_training():
    from model.train_model import train, derive_probas, build_features, load_data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"データファイルが見つかりません: {DATA_PATH}")
    bundle = train(DATA_PATH, out_path=None, verbose=False)
    if "model" not in bundle:
        raise ValueError("学習結果にmodelが含まれていません")

    # 複勝率・勝率が導けるか、また「勝率が複勝率を上回る」矛盾が
    # 起きていないかも確認する(以前このバグを踏んだことがあるため)
    df = load_data(DATA_PATH)
    X, _ = build_features(df, encoders=bundle["encoders"])
    top3_proba, win_proba = derive_probas(bundle["model"], X)
    violations = (win_proba > top3_proba + 1e-9).sum()
    if violations > 0:
        raise ValueError(f"勝率が複勝率を上回っている行が{violations}件あります")


def check_schema():
    import pandas as pd
    from data_collector import CSV_COLS
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV_COLSに定義された列が無い: {missing}")
    if len(df) == 0:
        raise ValueError("データが0行です")
    required_never_null = ["race_id", "venue", "distance", "track_type", "horse_name", "finish_rank"]
    for col in required_never_null:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"{col}列にNaNが{n_null}件あります(この列は必須のはず)")


def check_no_duplicate_races():
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    if "horse_name" not in df.columns or len(df) == 0:
        return
    race_horses = df.groupby("race_id")["horse_name"].apply(lambda s: frozenset(s.dropna())).to_dict()
    seen = {}
    dups = []
    for rid, horses in race_horses.items():
        if not horses:
            continue
        for other_rid, other_horses in seen.items():
            if not other_horses:
                continue
            overlap = horses & other_horses
            smaller = min(len(horses), len(other_horses))
            if smaller > 0 and len(overlap) / smaller >= 0.8 and abs(len(horses) - len(other_horses)) <= 2:
                dups.append((rid, other_rid))
        seen[rid] = horses
    if dups:
        raise ValueError(f"出走馬がほぼ一致するrace_idの組み合わせが見つかりました(重複の疑い): {dups}")


def check_no_race_id_collision():
    """同じrace_idなのに、実際には複数の別レースが混在しているケースを検出する。

    手動でrace_idの開始番号を指定した際に、複数のレースで同じ番号から
    始めてしまうと発生する。「重複レースの検出」(check_no_duplicate_races)
    とは逆方向のチェック: あちらは「別IDなのに中身が同じ」を検出するが、
    こちらは「同じIDなのに中身が別」を検出する。

    判定基準:
    - 同じrace_idの中に finish_rank == 1(1着)の行が2件以上ある
      → 1レースに1着馬は1頭のはずなので、複数レースが混ざっている強い証拠
    - 同じrace_idの出走頭数が18頭を超えている
      → JRAの1レース最大出走頭数(18頭)を超えることは無い
    """
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    if "race_id" not in df.columns or "finish_rank" not in df.columns or len(df) == 0:
        return
    collisions = []
    for rid, group in df.groupby("race_id"):
        n_horses = len(group)
        n_winners = (group["finish_rank"] == 1).sum()
        if n_winners >= 2:
            collisions.append((rid, "1着馬が複数", int(n_winners)))
        elif n_horses > 18:
            collisions.append((rid, "頭数が18を超過", int(n_horses)))
    if collisions:
        raise ValueError(
            f"同じrace_idに別レースが混在している疑いがあります(race_id, 理由, 件数): {collisions}"
        )


def main():
    check("主要モジュールのimport", check_imports)
    check("データのスキーマ", check_schema)
    check("重複レースの検出", check_no_duplicate_races)
    check("race_id衝突の検出", check_no_race_id_collision)
    check("モデルの学習", check_training)

    print()
    if errors:
        print(f"❌ {len(errors)}件のチェックに失敗しました: {errors}")
        sys.exit(1)
    else:
        print("✅ 全てのチェックに合格しました")
        sys.exit(0)


if __name__ == "__main__":
    main()
