# CLAUDE.md

このファイルは、このリポジトリで作業する際にClaude(Claude Code含む)が参照するためのプロジェクト情報です。

## プロジェクト概要

競馬予想AI。Streamlit + scikit-learnで構築し、GitHub + Streamlit Community Cloudで公開している個人開発アプリ。
netkeibaの結果ページ・出馬表ページのデータを手動または自動取得で蓄積し、GradientBoostingClassifierで
「3着以内に入るか(複勝)」「1着になるか(勝率)」を予測する。

ユーザーはプログラミング未経験(コードは書けない)。iPad + Chrome、iPhoneの両方で操作する。
そのため**このリポジトリのコードは常にClaude(このAI)が直接編集し、ユーザーはGitHub上に
コピー&ペーストで反映するだけ**という運用になっている。ユーザーに「このコードを実行して」
のような指示は出さない。

## デプロイ構成

- リポジトリ: `mori0227mitsu-lab/Keiba-ai` (Public, branch: `main`)
- ホスティング: Streamlit Community Cloud (`keiba-ai-hddvdkauy7vxtqxbdetg9t.streamlit.app`)
- GitHubにpushすると自動でリビルド・再デプロイされる
- Streamlit CloudのSecretsに `github_token` / `github_repo` / `github_branch` を設定済み
  (アプリ自身がGitHub API経由でCSVを読み書きするため)
- **Claude(このAI)はGitHubに直接コミットする手段を持たない**。作業のたびに、修正したファイルを
  `present_files`で渡し、ユーザーがGitHubのWeb UIでコピペして反映する運用。反映後は「リブートしてください」
  と伝える(Streamlit Cloudはアプリの再起動が必要な場合がある)。

## ファイル構成

```
Keiba-ai/
├── app.py                   # Streamlitアプリ本体(UI)
├── requirements.txt         # 依存ライブラリ
├── course_info.py           # 競馬場の固定知識(直線の長さ、右左回り、坂の有無など)
├── race_class.py            # レース格(新馬/未勝利/1勝クラス/オープン/G1-G3)の判定
├── data_collector.py        # netkeibaのテキスト・URLからのデータ抽出モジュール
├── github_sync.py           # GitHub API経由でのCSV読み書き・重複チェック
├── .streamlit/config.toml   # テーマ設定
├── data/
│   └── dummy_races.csv      # 蓄積している実データ本体(CSV_COLS形式)
└── model/
    └── train_model.py       # 特徴量エンジニアリング・学習・バックテスト・買い目期待値計算
```

**注意**: `model.joblib`はコミットしない。アプリ起動時に`train_model.train()`が自動学習して作る
(Streamlitの`@st.cache_resource`でキャッシュされる)。

## データスキーマ(CSV_COLS)

`data/dummy_races.csv` の列順:

```
race_id, race_date, venue, distance, track_type, condition, straight_length,
day_bias, race_class, class_level, horse_num, horse_name, waku, sex, age,
jockey, trainer, running_style, weight_carry, horse_weight, weight_diff,
prev_rank, rest_weeks, popularity, odds, finish_rank,
time_sec, agari_3f, corner_pos
```

- `race_date`: "YYYY-MM-DD" 文字列。無い場合(古いデータ)は空欄または`"0"`。
- `race_id`: 数値だが**時系列順とは限らない**(後から日付ありで別レースを追加すると番号が前後する)。
  時系列判定には必ず `race_date` を優先し、無い場合だけ `race_id` にフォールバックする
  (`add_chronological_sort_key`参照)。
- `prev_*` 列(prev_rank, prev_time_sec, prev_class_level, prev_race_time_score,
  prev_field_strength_note, prev_stretch_out_note, rest_weeks)は**保存時に空/0で入れておき、
  学習時(`fill_prev_from_history`)に馬名ベースで自動計算し直す**。CSVに保存された値そのものは
  信用しない(常に再計算される)。

## 特徴量エンジニアリングの全体像(model/train_model.py)

`load_data()` → `fill_prev_from_history()` → `compute_horse_course_aptitude()` →
`compute_horse_distance_aptitude()` の順で、生データに以下を付与してから学習する。

| 機能 | 関数 | 概要 |
|---|---|---|
| 展開評価 | `compute_pace_note` | 強い勝ち方/展開不利をレース内の脚質・上がり・着順から判定 |
| レースタイム水準 | `compute_race_time_level` | 同条件(距離+コース種別)の平均タイムとの差 |
| 相手のレベル評価 | `compute_field_strength_note` | 僅差(1.0秒以内)で先着した馬**2頭以上**が後日掲示板(5着以内)に載っていれば高評価。1頭だけでは判定しない(緩すぎるとの指摘で厳格化した経緯あり) |
| 距離延長候補 | `compute_stretch_out_note` | 短距離(≤1400m)・後方25%通過・上がりがレース平均よりはっきり速い・掲示板外、の5条件全部を満たす場合のみ |
| 右左回り/坂/距離適性 | `compute_horse_course_aptitude` / `compute_horse_distance_aptitude` | 馬ごとに、条件が合う/合わないでの複勝率差(閾値0.34)で判定。**必ず「そのレースより前」のデータだけを使う(データリーク防止)** |
| レース格 | `race_class.detect_race_class` | ヘッダーテキストから新馬〜G1を正規表現で判定。**「L」判定は`\(L\)`のようにカッコ必須**(単独の"L"に誤反応した実バグあり) |

FEATURE_COLS / CATEGORICAL_COLS は `model/train_model.py` 冒頭で一元管理。新しい特徴量を追加する際は
両方のリストと、`build_features()`内の欠損時デフォルト値補完(既存データに新列が無くてもエラーにならないように)を
必ずセットで追加すること。

## データ収集(重複防止まわりが最重要・過去に何度もバグを踏んだ箇所)

`github_sync.find_existing_race_ids()` が「既にCSVに入っているレースかどうか」を判定する。

**現在のロジック**: 開催場・距離・コース種別が一致し、かつ
1. 両方に開催日があれば日付が一致するか
2. 開催日が無ければ、**出走馬名の重なりが80%以上、かつ頭数差が2頭以内**

の場合だけ同一レースとみなして上書きする。

### 踏んだバグの履歴(同じ轍を踏まないこと)
1. 開催場+距離+コース種別だけで判定 → 偶然同条件の別レースを誤って同一視し、データを破壊した実例あり
2. 列を揃える際の欠損値デフォルトが数値`0`だったため、`race_date`のような文字列列も`"0"`で埋まってしまった
3. 出走馬名の重なり判定を「50%以上」にしたら、同じ場所・距離で複数の馬が2走目として再出走しているだけの
   別レースを誤検知した → 「頭数差2頭以内 かつ 80%以上」に厳格化して解決

## URL直接取得機能(実験的)

`data_collector.fetch_and_parse_netkeiba_result(url, race_id, race_date)` で、netkeibaの結果ページURLから
直接構造化データを取得できる(コピペ不要)。

- **スマホ版(`race.sp.netkeiba.com`)とPC版(`race.netkeiba.com`)はレイアウトが全く違う**。
  スマホ版は `12R 3歳以上1勝クラス16:30 ダ1150m(右) 14頭 晴 良` のように1行に情報が詰まっており、
  既存のパーサーでは読めない。そのため `normalize_netkeiba_url()` で必ずPC版URLに変換してから取得する。
- ページ内の複数の`<table>`から本物の結果テーブルを`_find_race_table()`でキーワードスコアリングして選ぶ
  (「過去のレース結果」のような紛らわしい表が同じページ内にあるため、単純に最初の表を使うと誤爆する)。
- 空セル(着差が無い1着馬など)を詰めて除去すると後続列がズレるため、**空セルも位置を保持したまま残す**。
- race_class判定にページ全体のテキストを渡すと、無関係な箇所の「L」の一文字に誤反応するため、
  開催情報の前後(`class_context`)に範囲を絞って渡す。

## 買い目期待値計算

`model/train_model.py`に、勝率(`win_model`、finish_rank==1を予測する2つ目のモデル)と
ハーヴィルの公式(`quinella_proba` / `wide_proba` / `trio_proba` / `trifecta_proba`)を使った
期待値計算がある。app.py側では、印がついた馬から**自由に**組み合わせを選べるようにしてある
(◎軸固定ではない)。

## UIの既知の注意点(Streamlitのクセ)

- ボタン(`st.button`)の中で計算した結果を、その後に表示する別のウィジェット(セレクトボックス等)を
  ユーザーが操作すると、ボタンの条件が再度Falseになり画面全体が消える。**予測結果のような
  「後続の操作で消えてほしくない状態」は必ず`st.session_state`に保存し、ボタンの外(`if "xxx" in
  st.session_state:`)で表示する**。
- テキスト入力欄をボタン処理後にクリアしたい場合、`st.session_state`に直接書き込むとエラーになるため、
  「バージョンカウンターをインクリメントして、keyが違う新しいウィジェットを生成する」方式を使う
  (`raw_paste_version`, `collect_paste_version`, `result_urls_version`などが実例)。

## 今後の作業で意識すること

- 新しい特徴量を追加したら、必ず: (1) FEATURE_COLS/CATEGORICAL_COLSに追加 (2) `fill_prev_from_history`で
  前走への引き継ぎが必要か検討 (3) `build_features`に欠損時デフォルト値を追加 (4) app.py側の
  「馬名から前走成績を自動入力」「予測実行」両方に同じデフォルト値初期化を追加、を漏れなく行う。
- ユーザーはコードを書けないため、**変更は必ず動作確認(可能な範囲でのユニットテスト)をしてから
  ファイルを渡す**。特にGitHub Actionsのような自動テストは無いので、Claudeがその場でbashを使って
  検証するのが唯一の品質担保。
- ユーザーはnote(https://note.com)で「蜜馬」名義で開発日記・予想記事を書いている。アプリの新機能や
  発見(例: 距離延長候補馬)は、note記事のネタとして一緒に文章化することもある。
