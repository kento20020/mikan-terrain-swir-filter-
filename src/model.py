"""Stage-1 みかん園地検出（Random Forest）+ Stage-2 出荷量回帰（XGBoost）。"""

from __future__ import annotations

import pickle
import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from xgboost import XGBRegressor

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

BASE = Path(__file__).parent.parent
MODELS_DIR = BASE / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_NON_FEATURE_COLS = {"polygon_id", "label", "geometry", "land_type", "area_m2"}


# ── Stage-1: みかん園地分類 ───────────────────────────────────────────────

def load_stage1_data(features_path: Path) -> tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    """features_s2.geojson を読み込み (gdf, X, y) を返す。"""
    gdf = gpd.read_file(features_path)
    feature_cols = [c for c in gdf.columns if c not in _NON_FEATURE_COLS and c != "geometry"]
    X = gdf[feature_cols].fillna(0).values.astype(np.float32)
    y = gdf["label"].values.astype(int)
    return gdf, X, y


def spatial_split(gdf: gpd.GeoDataFrame, test_frac: float = 0.2) -> tuple:
    """
    緯度順で末尾 test_frac をテストセットとする空間分割。
    ランダム分割では空間的自己相関によりリーケージが生じるため使用しない。
    """
    feature_cols = [c for c in gdf.columns if c not in _NON_FEATURE_COLS and c != "geometry"]
    wgs84 = gdf.to_crs("EPSG:6677")  # 平面直角座標系で重心計算
    sort_idx = wgs84.geometry.centroid.y.argsort().values
    n_test = int(len(gdf) * test_frac)
    test_idx  = sort_idx[-n_test:]
    train_idx = sort_idx[:-n_test]

    X_all = gdf[feature_cols].fillna(0).values.astype(np.float32)
    y_all = gdf["label"].values.astype(int)
    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]


def train_stage1_rf(features_path: Path) -> tuple[RandomForestClassifier, dict]:
    """
    Random Forestでみかん園地を検出する。
    ラベル仮定: 八幡浜市の南向き斜面の畑（land_type=200）はみかん園地として扱う。
    作物種別コードは筆ポリゴンに存在しない。混作地域では成立しない。
    """
    gdf, X, y = load_stage1_data(features_path)
    feature_cols = [c for c in gdf.columns if c not in _NON_FEATURE_COLS and c != "geometry"]

    X_train, X_test, y_train, y_test = spatial_split(gdf)

    rf = RandomForestClassifier(
        n_estimators=CONFIG["model"]["n_estimators_rf"],
        oob_score=True,
        class_weight="balanced",
        random_state=CONFIG["model"]["random_state"],
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    report = classification_report(y_test, rf.predict(X_test), output_dict=True)
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print(classification_report(y_test, rf.predict(X_test)))

    # 特徴量重要度
    importances = dict(zip(feature_cols, rf.feature_importances_))
    top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    print("特徴量重要度 TOP10:")
    for name, imp in top10:
        print(f"  {name}: {imp:.4f}")

    out_path = MODELS_DIR / "stage1_rf.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": rf, "feature_cols": feature_cols}, f)
    print(f"保存: {out_path}")

    return rf, report


def estimate_mikan_area(rf: RandomForestClassifier, features_path: Path) -> float:
    """Stage-1 RF でみかんと判定されたポリゴンの面積合計（ha）を返す。"""
    gdf, X, _ = load_stage1_data(features_path)
    probs = rf.predict_proba(X)[:, 1]
    mikan_mask = probs >= 0.55
    area_ha = gdf.loc[mikan_mask, "area_m2"].sum() / 10000 if "area_m2" in gdf.columns else float("nan")
    print(f"Stage-1推定みかん面積: {area_ha:.1f} ha ({mikan_mask.sum()}/{len(gdf)}ポリゴン)")
    return area_ha


# ── Stage-2: 出荷量回帰 ──────────────────────────────────────────────────

def load_stage2_data(features_csv: Path, labels_csv: Path) -> pd.DataFrame:
    """Landsat特徴量と出荷量ラベルをyearで結合する。"""
    feat_df  = pd.read_csv(features_csv)
    label_df = pd.read_csv(labels_csv)
    df = feat_df.merge(label_df, on="year", how="inner").dropna(subset=["shipment_t"])
    print(f"Stage-2データ: {len(df)}件 ({df['year'].min()}-{df['year'].max()})")
    direct = df[~df["is_proxy"]]
    proxy  = df[df["is_proxy"]]
    print(f"  直接測定年: {len(direct)}件, 疑似ラベル年: {len(proxy)}件")
    return df


def train_stage2_xgb(
    df: pd.DataFrame,
    stage1_area_ha: float | None = None,
) -> tuple[XGBRegressor, dict]:
    """
    XGBoostで出荷量を回帰する。
    Stage-2疑似ラベル警告: 2007年以降の出荷量は愛媛県計×31%の推計値。
    実測値ではない。誠実なMAPE評価は直接測定年（is_proxy=False）のみで行う。
    """
    exclude = {"year", "shipment_t", "is_proxy"}
    feature_cols = [c for c in df.columns if c not in exclude]

    if stage1_area_ha is not None and "stage1_area_ha" not in df.columns:
        df = df.copy()
        df["stage1_area_ha"] = stage1_area_ha
        feature_cols.append("stage1_area_ha")

    X = df[feature_cols].fillna(0).values
    y = df["shipment_t"].values

    model = XGBRegressor(
        n_estimators=CONFIG["model"]["n_estimators_xgb"],
        max_depth=CONFIG["model"]["max_depth_xgb"],
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=CONFIG["model"]["random_state"],
    )

    # LOOCV（N≈24の少数データ）
    loo = LeaveOneOut()
    loocv_preds = cross_val_predict(model, X, y, cv=loo)

    # 分離評価
    is_proxy = df["is_proxy"].values
    years    = df["year"].values
    direct_mask = ~is_proxy

    def mape(actual, pred):
        return float(np.mean(np.abs((actual - pred) / actual)) * 100)

    direct_mape = mape(y[direct_mask], loocv_preds[direct_mask]) if direct_mask.any() else float("nan")
    proxy_mape  = mape(y[~direct_mask], loocv_preds[~direct_mask]) if (~direct_mask).any() else float("nan")
    all_mape    = mape(y, loocv_preds)

    print(f"\n=== Stage-2 評価（LOOCV）===")
    print(f"MAPE（直接測定年 {years[direct_mask].tolist()}）: {direct_mape:.1f}%  ← 主要指標")
    print(f"MAPE（疑似ラベル年）: {proxy_mape:.1f}%  ※実測値ではない")
    print(f"MAPE（全年）: {all_mape:.1f}%")

    # 最終モデルを全データで学習
    model.fit(X, y)

    out_path = MODELS_DIR / "stage2_xgb.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"保存: {out_path}")

    metrics = {
        "mape_direct": direct_mape,
        "mape_proxy":  proxy_mape,
        "mape_all":    all_mape,
        "loocv_preds": loocv_preds,
        "years":       years,
        "actual":      y,
        "is_proxy":    is_proxy,
    }
    return model, metrics


if __name__ == "__main__":
    feat_s2 = BASE / "data" / "processed" / "features_s2.geojson"
    feat_lc  = BASE / "data" / "processed" / "features_landsat.csv"
    labels   = BASE / "data" / "raw" / "stats" / "yawatahama_shipment.csv"

    # Stage-1
    if feat_s2.exists():
        print("=== Stage-1: みかん園地検出 ===")
        rf, report = train_stage1_rf(feat_s2)
        area_ha = estimate_mikan_area(rf, feat_s2)
    else:
        print(f"警告: {feat_s2} が存在しません。Phase 2を先に実行してください。")
        area_ha = None

    # Stage-2
    if feat_lc.exists() and labels.exists():
        print("\n=== Stage-2: 出荷量回帰 ===")
        df = load_stage2_data(feat_lc, labels)
        xgb_model, metrics = train_stage2_xgb(df, stage1_area_ha=area_ha)
    else:
        missing = [p for p in [feat_lc, labels] if not p.exists()]
        print(f"警告: {missing} が存在しません。Phase 3-4を先に実行してください。")
