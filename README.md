# mikan-terrain-swir-filter-
ClaudeCodeに実装を丸投げした八幡浜市のみかんの出荷量を衛星画像から分析するモデル

zenn_article_draft.md
にはZennの記事を書かせてみた。
00_pipeline_overview.ipynb
00_pipeline_overview_executed.ipynb
にの実際にどのようなデータをどう処理したのかも書かせてある。


下記は読み込ませたプロンプトです。

# 八幡浜みかん産地 時系列分析プロジェクト

## このファイルの読み方

Claude Code はタスクを受け取る前にこのファイルを必ず全読みすること。
指示が曖昧なときは「## 応答ガイド」を参照して判断する。
新しいファイルを作る前に「## ディレクトリ構成」を確認する。

---

## プロジェクトの目的と核心的な問い

八幡浜市・西宇和地区のみかん産地崩壊を、衛星データ・公式農業統計・農家統計の
3つの視点から定量化する。

| 問い | 対応するデータ | 対応するSTEP |
|------|--------------|------------|
| 衛星面積 vs 統計面積の乖離は？ | Landsat + 作物統計 | STEP 3・6 |
| 耕作放棄はいつ・どこで起きたか | LandTrendr | STEP 5 |
| 農家数減少と面積減少のズレは？ | 農林業センサス + 衛星 | STEP 6 |
| 統計に見えない放棄地はどこか | 衛星 - 統計の差分 | STEP 6 |

---

## 実行環境

```
OS       : WSL2 Ubuntu 24.04
GPU      : RTX 3060 Ti（現時点では未使用。将来のCNN分類フェーズで活用）
Python   : 3.11（仮想環境 .venv）
GEE処理  : Google側クラウド（ローカルGPUは使わない）
Python処理: WSLローカル（STEP 4〜7）
```

GEE認証コマンド（WSL用・ブラウザなし）:
```bash
earthengine authenticate --auth_mode=gcloud
# 表示されたURLをWindowsブラウザで開き、コードをターミナルに貼る
```

---

## ディレクトリ構成

```
mikan-satellite-analysis/
├── CLAUDE.md                        ← このファイル
├── .env                             ← APIキー（Git管理外）
├── data/
│   ├── raw/
│   │   ├── estat/
│   │   │   ├── sakumotu_ehime.csv   # 作物統計（みかん結果樹面積・年別）
│   │   │   └── census_ehime.csv     # 農林業センサス（世帯数・従事者・高齢化率）
│   │   └── satellite/
│   │       ├── area_active_ha.csv   # GEEエクスポート：年別活性みかん面積
│   │       └── ndvi_monthly.csv     # GEEエクスポート：月別NDVI集計値
│   └── processed/
│       └── merged_timeseries.csv    # 全データ統合済み時系列
├── src/
│   ├── gee_utils.py                 # GEE操作の共通関数
│   ├── estat_utils.py               # e-Stat APIラッパー
│   ├── preprocess.py                # 前処理（マスク・補正・補間）
│   ├── timeseries.py                # STL分解・Mann-Kendall・変化点検出
│   ├── gap_analysis.py              # 乖離分析・農家統計との関係分析
│   └── visualize.py                 # 可視化の共通関数
├── outputs/
│   ├── figures/                     # グラフ・ダッシュボード（PNG）
│   └── maps/                        # LandTrendrマップ（GeoTIFF）
└── notebooks/
    ├── 01_gee_area.ipynb
    ├── 02_estat.ipynb
    └── 03_analysis.ipynb
```

---

## 実装状態

| ファイル | 状態 | 備考 |
|---------|------|------|
| `mikan_area_analysis.py` | 合成データで動作確認済み | `build_datasets()` を実データに差し替えれば使える |
| `mikan_yield_prediction.py` | 合成データで動作確認済み | Random Forest 収量予測 |
| `gee_mikan_longterm.py` | GEE認証済み環境で動作する | LandTrendr実装済み |
| `src/gee_utils.py` | 未作成 | |
| `src/estat_utils.py` | 未作成 | |
| `src/timeseries.py` | 未作成 | STL・Mann-Kendall |
| `src/gap_analysis.py` | 未作成 | 農家統計との関係分析 |
| `src/visualize.py` | 未作成 | |

---

## データスキーマ

### merged_timeseries.csv

```
year              : int    # 年（1990〜2024）
area_stat_ha      : float  # 農水省統計 みかん結果樹面積（ha）
area_sat_ha       : float  # 衛星由来 活性みかん園地面積（ha）
abandoned_ha      : float  # 耕作放棄推定 = area_stat - area_sat
abandoned_ratio   : float  # 耕作放棄率（%）= abandoned / area_stat × 100
ndvi_mean_summer  : float  # 7〜9月の平均NDVI
farm_households   : float  # 農家世帯数（センサス年のみ実測、他は補間）
farm_workers      : float  # 農業従事者数（同上）
elderly_ratio     : float  # 65歳以上従事者割合（%）
successor_ratio   : float  # 後継者あり農家割合（%）
area_per_hh_stat  : float  # 統計ベース 1世帯あたり担当面積（ha）
area_per_hh_sat   : float  # 衛星ベース 1世帯あたり担当面積（ha）
interpolated      : bool   # True = 補間値、False = 実測値
```

センサス年は 2000・2005・2010・2015・2020 のみ実測値。
それ以外の年は線形補間で埋めたうえで `interpolated=True` フラグを付ける。

---

## 分析パイプライン（7ステップ）

### STEP 1｜データ収集

**GEE側（衛星）**
- Landsat 5/7/8/9 を 1990〜2024 年で取得（7〜9月中央値合成）
- 雲フィルタ: `CLOUD_COVER < 30`（Landsat 7は20%）
- AOI: `ee.Geometry.Rectangle([132.35, 33.40, 132.60, 33.55])`
- エクスポート先: Google Drive → `data/raw/satellite/`

**e-Stat側（統計）**
- 作物統計（みかん結果樹面積）: statsId `0003313868`
- 農林業センサス（農家世帯数・従事者数）: statsId `0003076020`
- APIキーは `.env` の `ESTAT_API_KEY` から読む

**欠損対応ルール**
- 梅雨期（6〜7月）はSentinel-2が取れない → 7〜9月の晴天画像のみ使う
- 1990〜2000年代はLandsat 7のSLC-offノイズあり → 雲フィルタを厳しくする
- それでも取れない年はNaNのまま記録し、補間は STEP 2 で行う

---

### STEP 2｜前処理

- 大気補正: Landsat C02 SR プロダクトをそのまま使う（補正済み）
- スケール変換: `DN × 0.0000275 - 0.2` で反射率に変換
- 地形補正: 段々畑の急傾斜（10〜40度）によるNDVI低下を補正
  - ALOS DEM（5m）で傾斜角・斜面方位を計算
- 筆ポリゴンマスク: みかん農地に絞り込む（森・住宅地を除外）

---

### STEP 3｜時系列データ構築（GEE）

植生指数を計算して年別CSVに出力する。

| 指数 | 用途 |
|------|------|
| NDVI | 植生量・生育状況の基本指標 |
| NBR | 枯死・耕作放棄に敏感。LandTrendrに使う |
| EVI | 高密度植生向け（段々畑の密植みかんに有効） |

**みかん園地の識別条件（常緑マスク）**
```
夏NDVI（7〜9月）> 0.50    # 夏に活性
冬NDVI（12〜2月）> 0.42   # 冬も枯れない（常緑）
夏冬差 < 0.15              # 季節変動が小さい（落葉樹は0.3以上）
筆ポリゴン（農地）内       # 農地以外を除外
```

この3条件ANDのピクセルだけを「活性みかん園地」とみなす。
常緑特性がみかんと森林を区別する最大のシグナル。

---

### STEP 4｜STL分解（Python）

`statsmodels.tsa.seasonal.STL` を使う。

- 対象: 月別NDVI時系列 / 年別面積時系列
- `period=12`（月別）または `period=5`（年別）
- `robust=True`（外れ値に強いモード）

**なぜ必要か**
みかんのNDVIには毎年「夏高く冬低い」季節変動が乗っている。
これを除去しないと衰退トレンドが見えない。
STEP 5以降はSTLのトレンド成分だけを使う。

---

### STEP 5｜トレンド検定・変化点検出（Python + GEE）

#### Mann-Kendall 検定
- ライブラリ: `pymannkendall`
- 対象: 衛星面積・統計面積・農家世帯数・農業従事者数の4系列
- 出力: p値・Sen's slope（年間変化量 ha/年、人/年）
- 判定基準: p < 0.05 を「有意なトレンドあり」とする

#### LandTrendr（GEE）
- 対象バンド: NBR（耕作放棄で急落する）
- 出力: 変化年（LTR_yod）・変化量（LTR_mag）・持続期間（LTR_dur）
- 耕作放棄フィルタ: `LTR_mag < -150` かつ `LTR_dur <= 3`（急落）

#### 変化点の農家統計との照合
- LandTrendrで検出した変化年を農林業センサス年（2000/05/10/15/20）と突き合わせる
- 「面積の急落年」と「農家数が大きく減った5年間」がどれだけずれているかを記録する
- これがこのプロジェクトで最も重要な発見になる可能性がある

---

### STEP 6｜乖離分析・農家統計との関係分析（Python）

農家統計との関係分析のメインSTEP。4つの分析を行う。

#### ① 時系列の正規化と重ね合わせ
- 4系列（衛星面積・統計面積・農家世帯数・農業従事者数）を2000年基準の変化率（%）に統一
- 単位が違う系列を同じグラフに乗せるための前処理
- 傾きの差が「誰が一番速く減っているか」を示す

#### ② 相関分析
センサス年（2000/05/10/15/20）のみを使う。

| 相関ペア | 意味 |
|----------|------|
| 農家世帯数 × 衛星面積 | 人が減れば実際の管理面積も減る |
| 農家世帯数 × 統計面積 | 統計には耕作放棄地が残るため乖離が出る |
| 農家世帯数 × 農業従事者数 | 連動の確認 |

Pearson相関係数 + 散布図 + 回帰直線で可視化する。

#### ③ 乖離率の算出と推移
```
乖離率（%）= (area_stat_ha - area_sat_ha) / area_stat_ha × 100
```
農家世帯数の減少ペースと乖離率の拡大ペースを比べて「タイムラグ」を測る。
乖離率が15%を超えたら STEP 5 のLandTrendrで地域を特定する。

#### ④ 1世帯あたり担当面積
```
area_per_hh_stat = area_stat_ha / farm_households  # 統計ベース
area_per_hh_sat  = area_sat_ha  / farm_households  # 衛星ベース（実態）
```
センサス年ごとに棒グラフで比較する。
両者の差が大きいほど「過大な負担を見かけ上担っている」状態を示す。

---

### STEP 7｜可視化・レポート生成（Python）

`outputs/figures/` に以下を生成する。

| ファイル名 | 内容 |
|-----------|------|
| `01_area_timeseries.png` | 衛星面積・統計面積・農家統計の長期推移 |
| `02_stl_decomposition.png` | STL分解の3成分 |
| `03_gap_ratio.png` | 乖離率（耕作放棄推定率）の年推移 |
| `04_correlation_scatter.png` | 農家世帯数 vs 各面積の散布図 |
| `05_area_per_hh.png` | 1世帯あたり担当面積の比較 |
| `06_normalized_change.png` | 2000年基準の正規化変化率比較 |
| `07_annual_change_rate.png` | 年別面積変化量 |

`outputs/maps/` に以下を生成する。

| ファイル名 | 内容 |
|-----------|------|
| `lt_abandonment_yod.tif` | LandTrendrの変化年マップ（GeoTIFF） |
| `lt_abandonment_mag.tif` | 変化量マップ |

---

## 応答ガイド

### タスクの種類と対応先

| ユーザーの発言 | やること |
|--------------|---------|
| 「GEEで〜を取得して」 | `src/gee_utils.py` に関数を追加 |
| 「e-Statから〜を取ってきて」 | `src/estat_utils.py` に関数を追加 |
| 「前処理して」「マスクして」 | `src/preprocess.py` を確認・編集 |
| 「STL分解して」「トレンドを見たい」 | `src/timeseries.py` を確認・編集 |
| 「農家数との関係を分析して」 | `src/gap_analysis.py` を確認・編集 |
| 「グラフを作って」「可視化して」 | `src/visualize.py` を確認・編集 |
| 「実データに差し替えて」 | `mikan_area_analysis.py` の `build_datasets()` を編集 |
| 「LandTrendrを動かして」 | `gee_mikan_longterm.py` を確認・実行 |
| 「どこまで進んでいるか」 | 「## 実装状態」テーブルを読んで現状を報告 |
| 「〜が動かない」「エラーが出た」 | エラーメッセージを読んで原因を特定し修正案を提示 |

### 作業前に必ず確認すること

1. 対象ファイルが「## ディレクトリ構成」に存在するか
2. 「## 実装状態」で未作成かどうか確認
3. 依存するデータが `data/raw/` に存在するか
4. 合成データか実データかを明示してから作業する

### 実装時のルール

- 言語: Python 3.11。型ヒントを必ず書く（例: `def func(x: pd.Series) -> float:`）
- 関数: 1関数1責務。処理の塊ごとに関数に分ける
- データ: pandas DataFrame を中心に扱う
- 保存: 図は `outputs/figures/`、マップは `outputs/maps/` に必ず保存
- コメント: 農業・衛星のドメイン知識を日本語で説明するコメントを書く
- エラー処理: GEE APIは通信エラーが頻発するので `try/except` と再試行ロジックを入れる
- テスト: 合成データで動作確認してから実データに切り替える

### 農林業センサスの補間ルール

センサスは2000・2005・2010・2015・2020年の5時点しかない。
年次データが必要な分析では以下で補間する。

- デフォルト: 線形補間（`pd.DataFrame.interpolate`）
- センサス年の値: そのまま使う
- `interpolated` カラムに `True/False` を付けて補間か実測かを区別する
- 外挿（2021〜2024年）: Sen's slopeから線形外挿するが「推計値」と明示する

### エラーが起きやすいポイント

| 場所 | よくあるエラー | 対処 |
|------|--------------|------|
| GEE認証 | WSLでブラウザが開かない | `--auth_mode=gcloud` を使う |
| GEE処理 | `Computation timed out` | `bestEffort=True` を追加、scaleを上げる |
| GEE処理 | `User memory limit exceeded` | AOIを小さくする |
| e-Stat API | 429 Too Many Requests | `time.sleep(1)` を挟む |
| pandas | センサス年以外がNaN | 補間前か確認。意図的なNaNは `.dropna()` しない |
| matplotlib | 日本語フォント化け | `rcParams['font.family'] = 'IPAGothic'` を設定 |

---

## よく使うコマンド

```bash
# 仮想環境の有効化
source ~/mikan-satellite-analysis/.venv/bin/activate

# パッケージインストール（初回）
pip install earthengine-api geemap \
            pandas numpy matplotlib seaborn scipy \
            statsmodels pymannkendall \
            scikit-learn xgboost \
            geopandas rasterio python-dotenv

# GEE認証（WSL環境）
earthengine authenticate --auth_mode=gcloud

# 分析実行（現在は合成データ版）
python src/mikan_area_analysis.py

# LandTrendr実行（GEE接続必須）
python gee_mikan_longterm.py

# matplotlib フォント確認
python -c "import matplotlib.font_manager as fm; print([f.name for f in fm.fontManager.ttflist if 'IPA' in f.name])"
```

---

## 参考リンク

| リソース | URL |
|---------|-----|
| Google Earth Engine | https://earthengine.google.com |
| geemap | https://geemap.org |
| e-Stat API | https://api.e-stat.go.jp |
| 農水省 筆ポリゴン | https://www.maff.go.jp/j/tokei/porigon/ |
| 農林業センサス | https://www.e-stat.go.jp |
| pymannkendall | https://pypi.org/project/pymannkendall/ |
| statsmodels STL | https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html |
| JAXA Earth API | https://data.earth.jaxa.jp |
