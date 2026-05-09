# 衛星画像×政府統計で八幡浜みかんの出荷量を予測する

## はじめに

「衛星から農作物の出荷量を予測できないか」というアイデアを実際に試してみた記録です。

対象は愛媛県八幡浜市の温州みかん。日本有数のみかん産地で、斜面みっしりのみかん畑が広がる地域です。使ったデータは以下の3種類だけです。

- **MAFF 筆ポリゴン**（農地の区画情報）
- **e-Stat API**（政府統計の出荷量データ）
- **Google Earth Engine**（Sentinel-2・Sentinel-1・Landsat衛星）

結果は **MAPE 12.5%**（直接測定年のみの評価）。実装のなかで踏んだバグや設計上の罠がいくつかあったので、その記録も含めてまとめます。

---

## 構成

```
yawatahama-mikan/
├── config.toml
├── src/
│   ├── fude.py          # 筆ポリゴン前処理
│   ├── gee_features.py  # Sentinel-2/S1/地形 99バンド特徴量
│   ├── gee_landsat.py   # Landsat 年次特徴量
│   ├── estat_shipment.py # e-Stat 出荷量統計
│   ├── model.py         # Stage-1 RF + Stage-2 XGBoost
│   └── evaluate.py      # 評価・可視化
└── notebooks/
    └── 00_pipeline_overview.ipynb
```

---

## 設計: なぜ2層構造にしたか

最初に直面した問題は **時系列のギャップ** です。

```
1993   ...   2006   2007   ...   2015   ...   2023
│←── e-Stat 市区町村出荷量データ ──►│ (ここで終了)
│←──────── Landsat (1993〜) ─────────────────────►│
                               │←── Sentinel-2 ──►│
```

Sentinel-2（10m解像度）は2015年以降しか使えません。一方、e-Statで八幡浜市の出荷量を**市区町村単位**で取得できるのは2006年が最後です。両者が重なる期間が存在しないため、「Sentinel-2で回帰する」は最初から無理でした。

そこで **2段階パイプライン** にしました。

| Stage | 目的 | 入力 | ラベル |
|-------|------|------|--------|
| Stage-1 | みかん園地の空間分類 | Sentinel-2/S1 99バンド × ポリゴン | 筆ポリゴン地目コード |
| Stage-2 | 出荷量の時系列回帰 | Landsat 年次特徴量 | e-Stat 出荷量（t） |

Sentinel-2は空間分解能（10m）が高いので「どこがみかん畑か」の分類に使い、Landsatは時系列が長い（1993〜）ので「その年の収量はどうだったか」の回帰に使う分担です。

---

## データソース詳解

### A. MAFF 筆ポリゴン

農林水産省が整備した農地の区画ポリゴン（GeoJSON）です。[open.fude.maff.go.jp](https://open.fude.maff.go.jp) からユーザー登録後にダウンロードできます。

**重要な制約**: 筆ポリゴンには「地目コード」はあっても **作物種別コードはありません**。

| 地目コード | 意味 | 使い方 |
|-----------|------|--------|
| 200 | 畑 | みかん候補 → label=1 |
| 100 | 田 | 確実な非みかん → label=0 |

「八幡浜の畑 ≈ みかん」という地域特性に頼った仮定です。混作地域では成立しません。

前処理の流れ：

```python
# fude.py の処理フロー
37,716ポリゴン（生データ）
  ↓ 地目フィルタ（200=畑 or 100=田）
37,716件
  ↓ 面積フィルタ（≥500m²）
19,258件
  ↓ 地形フラグ付与（SRTM: 南向き 10-40°傾斜）
  terrain_ok=1: 10,261件
  terrain_ok=0:  8,997件
```

地形フラグ付与で最初ハマりました（後述）。

### B. e-Stat API — なぜ3つの統計IDが必要か

e-Stat（政府統計の総合窓口）のREST APIを使って出荷量を取得します。ところが統計の整備状況が年代によってバラバラで、**3つの異なる統計IDを使い分ける**必要がありました。

```python
# 1993-2005年: 果樹累年統計（市区町村別）
GET /getStatsData?statsDataId=0003274240&cdArea=38204&cdCat01=120

# 2006年: 別の市区町村別統計（統計IDが変わっている）
GET /getStatsData?statsDataId=0003022267&cdCat01=346&cdCat02=004

# 2007年以降: 市区町村別データが存在しない → 愛媛県計で代替
GET /getStatsData?statsDataId=0003313868&cdCat01=170&cdArea=38000
# → 取得値 × 0.31（八幡浜の推計シェア）= 疑似ラベル
```

2007年以降は市区町村別の果樹統計が公開されていないため、愛媛県計に推計シェア31%を掛けて代用しています。このシェアは2005・2006年の実績から計算した値です。

```
2005年: 50,800 ÷ 164,900 ≈ 30.8%
2006年: 37,000 ÷ 124,100 ≈ 29.8%
平均 ≈ 31%
```

この疑似ラベル年（2007-2016）を `is_proxy=True` でマークし、**モデル評価では直接測定年（1993-2006）のMAPEのみを主要指標**とします。「参考値と主要指標を混ぜない」ことが誠実な評価の肝です。

```
最終データ: 24行
  直接測定: 14件（1993-2006）
  疑似ラベル: 10件（2007-2016）
  ※ Landsatデータがある年のみ使用 → 実際は13行（9直接 + 4疑似）
```

### C. Google Earth Engine

#### Sentinel-2/S1: 99バンド特徴量

Stage-1の分類に使うポリゴン単位の特徴量です。

| 種類 | バンド | 月数 | 計 |
|-----|--------|------|----|
| Sentinel-2 | NDVI, NDre, NDMI, NDWI, GNDVI | 12 | 60 |
| Sentinel-1 | VH, VV, VH/VV比 | 12 | 36 |
| SRTM地形 | elevation, slope, aspect | — | 3 |
| **合計** | | | **99** |

月次コンポジットはCloud Score+でマスクしてからmedian合成します。Sentinel-1はSARなので雲の影響がなく、梅雨期も安定して取れます。

```python
# Sentinel-2の月次コンポジット（Cloud Score+マスク版）
join = ee.Join.saveFirst("cs_image")
condition = ee.Filter.equals(leftField="system:index", rightField="system:index")
joined = ee.ImageCollection(
    join.apply(s2_col, cs_col, condition)
)
composite = joined.map(lambda img:
    img.updateMask(ee.Image(img.get("cs_image")).select("cs").gt(0.65))
       .multiply(0.0001)
).median()
```

#### Landsat: 年次特徴量

Stage-2の回帰に使う年単位の特徴量です。

```python
# 各年の特徴量（8種）
{
    "year": 2003,
    "ndvi_summer_mean": 0.653,  # 6-8月NDVI平均
    "ndvi_summer_std":  0.265,  # 空間的ばらつき
    "ndvi_winter":      0.149,  # 12-2月NDVI（常緑樹量）
    "ndmi_summer":      0.346,  # 夏の水分状態
    "area_sat_ha":   2213.2,    # スペクトルマスクで推定したみかん面積
    "area_delta_ha": -7961.6,   # 前年比変化
    "n_images":           5,    # 使用画像枚数
}
```

---

## 実装でハマったこと3選

### 1. `ee.Image.pipe()` は存在しない

GEEのImageオブジェクトはpandasの `pipe()` メソッドを持っていません。最初こう書いてエラーになりました。

```python
# ❌ NG: AttributeError
composite = (
    renamed
    .pipe(gee_utils.add_ndvi)
    .pipe(add_ndre)
    .pipe(gee_utils.add_ndmi)
)

# ✅ OK: 素直に連続代入
indices = gee_utils.add_ndvi(renamed)
indices = add_ndre(indices)
indices = gee_utils.add_ndmi(indices)
```

### 2. `ee.data.computeFeatures()` の `GEOPANDAS_GEODATAFRAME` フォーマットは存在しない

GEEのドキュメントを見て書いたコードです。

```python
# ❌ NG: このフォーマットは存在しない
result = ee.data.computeFeatures({
    'expression': reduced_fc,
    'fileFormat': 'GEOPANDAS_GEODATAFRAME'
})
```

Drive経由のエクスポートは遅い（分単位でジョブ完了待ち）ので避けたかったのですが、正解は `geemap` でした。

```python
# ✅ OK: geemapでDrive不要・即時取得
import geemap
result_gdf = geemap.ee_to_gdf(reduced_fc)
```

`geemap.ee_to_gdf()` は内部でGEEのAPIを叩いてGeoDataFrameに変換してくれます。19,258ポリゴンを500件ずつバッチ処理して合計40分弱で取得できました。

### 3. 地形フラグ付与で10MBペイロード制限に引っかかる

最初の実装は1ポリゴンずつGEEに投げていました。

```python
# ❌ NG: 19,258回のAPI呼び出し → 3時間以上かかる
for idx, row in gdf.iterrows():
    val = terrain_mask.reduceRegions(...).getInfo()
    gdf.loc[idx, 'terrain_ok'] = val
```

バッチ処理に変えたら今度は別エラー：

```
Request payload size exceeds the limit: 10485760 bytes.
```

500件ずつに分割して解決しました。

```python
# ✅ OK: 500件バッチ × 39回
BATCH = 500
for start in range(0, len(wgs84), BATCH):
    batch = wgs84.iloc[start:start + BATCH]
    features = [
        ee.Feature(ee.Geometry(row.geometry.__geo_interface__), {"idx": int(start + i)})
        for i, (_, row) in enumerate(batch.iterrows())
    ]
    fc = ee.FeatureCollection(features)
    result_info = terrain_mask.reduceRegions(
        collection=fc, reducer=ee.Reducer.first(), scale=30
    ).getInfo()
    # idxで結果を逆引き
    for feat in result_info.get("features", []):
        props = feat.get("properties", {})
        idx = props.get("idx")
        val = next((v for k, v in props.items() if k != "idx"), None)
        idx_to_flag[idx] = int(val or 0) if val is not None else 0
```

---

## モデル

### Stage-1: Random Forest（みかん園地分類）

```python
RandomForestClassifier(
    n_estimators=300,
    oob_score=True,
    class_weight='balanced',  # 畑:田 = 39:1 の不均衡対策
    n_jobs=-1,
)
```

**空間分割**: 緯度順で後ろ20%をテストセットにします。ランダム分割だと空間的自己相関でリーケージが生じるためです。

```python
wgs84 = gdf.to_crs("EPSG:6677")
sort_idx = wgs84.geometry.centroid.y.argsort().values
n_test = int(len(gdf) * 0.2)
test_idx  = sort_idx[-n_test:]   # 北部
train_idx = sort_idx[:-n_test:]  # 南部
```

**結果**: OOB Score 0.9934

ただしこれは「畑 vs 田を区別できている」スコアです。作物種別は識別できていません。

**特徴量重要度で見えたこと**:

| 順位 | 特徴量 | 意味 |
|-----|--------|------|
| 1 | GNDVI_03（3月） | 春先の葉面積指数 |
| 2 | elevation（標高） | みかん＝斜面、田＝低地 |
| 3 | NDVI_03（3月） | 春先の植生量 |

3月（冬明け直後）が最も重要という結果は直感的にも納得で、常緑のみかんと休耕田・落葉樹の差が最大になる時期です。

### Stage-2: XGBoost（出荷量回帰）

N=13という少数データなので過学習を抑える設定にしています。

```python
XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
)
```

評価はLOOCV（Leave-One-Out Cross-Validation）。1年を外してN-1年で学習→外した年を予測、を全年分繰り返します。

---

## 結果

```
=== Stage-2 LOOCV 評価 ===

MAPE（直接測定年 9件: 1994,1995,1997,2000-2004,2006）: 12.5%  ← 主要指標
MAPE（疑似ラベル年 4件: 2007,2009,2010,2014）         : 20.8%  ※参考値
MAPE（全13件）                                          : 15.0%

year  actual_t  pred_t  error_pct
1994     33200   41232      +24.2%  ← 最大誤差
1995     38200   42482      +11.2%
1997     48700   40266      -17.3%  ← 豊作年を過小評価
2000     39400   42513       +7.9%
2001     43200   40070       -7.2%
2002     39500   38128       -3.5%
2003     42400   39009       -8.0%
2004     45800   39267      -14.3%
2006     37000   43853      +18.5%
```

---

## 誤差の考察

モデルの予測レンジが **38,000〜44,000t** に収まっているのに対し、実測は **33,000〜49,000t** と幅が広い。典型的な「平均回帰」問題です。

**低値を過大推定**（1994: +24.2%、2006: +18.5%）と**高値を過小推定**（1997: -17.3%）が対称的に出ています。

根本的な原因は、Landsatが捉える「植生の平均状態」と、実際の出荷量を左右する要因のギャップです。

| 要因 | Landsatで見えるか |
|------|-----------------|
| 夏のNDVI・NDMI | ✅ 見える |
| **隔年結果性**（豊作→翌年不作のサイクル） | ❌ 見えない |
| 開花期の霜・台風被害 | △ 数週間後のNDVI低下で間接的に |
| 農家の出荷判断・価格要因 | ❌ 見えない |

みかんは **隔年結果性** が強い果樹で、豊作年の翌年は生理的に着果数が減ります。前年出荷量を特徴量に加えると改善できる可能性があります。

---

## 改善できそうなこと

### 前年出荷量（lag-1）の追加

```python
df['shipment_lag1'] = df['shipment_t'].shift(1)
# → 隔年サイクルを特徴量として与える
```

### AMeDAS気象データの統合

開花期（2-4月）の最低気温・降水量など、収量に直結する気象情報をLandsatの特徴量に追加する。

### N数の拡大

e-Stat以外の愛媛農業統計（県刊行物）から直接測定年を増やす。現状9件では XGBoost で学習できる上限に限界がある。

---

## まとめ

- **GEE × 筆ポリゴン × e-Stat** を組み合わせることで、フィールド調査なしで農業統計の予測パイプラインを構築できた
- Sentinel-2（2015〜）と市区町村統計（〜2006）の時系列ギャップを、**2層構造**（Sentinel-2で分類 / Landsatで回帰）で解決した
- 疑似ラベルと直接測定を **`is_proxy` フラグで明示的に分離**してMAPEを報告することが誠実な評価につながる
- 精度 **MAPE 12.5%**（直接測定9年）は目標の20%以下を達成。ただし隔年結果性・気象要因が未考慮で「平均回帰」が残る

---

## 使用ライブラリ

- `earthengine-api` + `geemap` — GEE操作・FeatureCollection→GeoDataFrame変換
- `geopandas` — 筆ポリゴンの地理空間処理
- `scikit-learn` — Random Forest・LOOCV
- `xgboost` — 出荷量回帰
- `tomllib`（Python 3.11+標準）— 設定ファイル読み込み（PyYAML不要）
- `requests` — e-Stat API呼び出し

---

## コード

GitHub: （リンク未設定）

Jupyter notebook（パイプライン全体の可視化付き解説）: `notebooks/00_pipeline_overview.ipynb`
