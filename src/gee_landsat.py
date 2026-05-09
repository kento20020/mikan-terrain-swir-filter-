"""Landsat年次特徴量取得: gee_utils.py の関数を再利用して1993-2016年の特徴量を生成する。"""

from __future__ import annotations

import sys
import time
import tomllib
from pathlib import Path

import pandas as pd

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import ee
import gee_utils  # 既存関数をすべて直接再利用

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_annual_landsat_features(year: int) -> dict | None:
    """
    指定年のLandsat衛星特徴量を計算する。
    compute_annual_mikan_area() を拡張し、回帰用特徴量ベクトルを返す。
    面積前年差（area_delta_ha）を含む（翌年計算時に前年データが必要）。
    """
    try:
        # gee_utils の内部関数を直接再利用
        summer_col = gee_utils._get_landsat_collection(
            year, CONFIG["landsat"]["cloud_cover"]
        ).map(gee_utils.add_ndvi).map(gee_utils.add_ndmi)

        if summer_col.size().getInfo() == 0:
            print(f"  {year}年 → スキップ（夏季画像なし）")
            return None

        winter_col = gee_utils._get_winter_collection(year)
        if winter_col.size().getInfo() == 0:
            print(f"  {year}年 → スキップ（冬季画像なし）")
            return None

        aoi = gee_utils.get_aoi()
        summer_median = summer_col.median()
        summer_ndvi   = summer_median.select("NDVI")
        summer_ndmi   = summer_median.select("NDMI")
        winter_ndvi   = winter_col.median().select("NDVI")
        n_images      = summer_col.size().getInfo()

        # NDVI統計（mean + std）
        stats = summer_ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=aoi, scale=30, bestEffort=True,
        ).getInfo()
        s_mean = stats.get("NDVI_mean", float("nan"))
        s_std  = stats.get("NDVI_stdDev", float("nan"))

        # 冬季NDVI
        w_info = winter_ndvi.reduceRegion(
            ee.Reducer.mean(), aoi, 30, bestEffort=True
        ).getInfo()
        w_mean = w_info.get("NDVI", float("nan"))

        # 夏季NDMI
        ndmi_info = summer_ndmi.reduceRegion(
            ee.Reducer.mean(), aoi, 30, bestEffort=True
        ).getInfo()
        ndmi_mean = ndmi_info.get("NDMI", float("nan"))

        # 品質チェック（compute_annual_mikan_area と同じ条件）
        diff_mean = s_mean - w_mean
        if w_mean > 0.55:
            print(f"  {year}年 → スキップ（冬NDVI={w_mean:.3f} > 0.55）")
            return None
        if diff_mean < 0.09:
            print(f"  {year}年 → スキップ（夏冬差={diff_mean:.3f} < 0.09）")
            return None

        # みかん推定面積
        terrain_mask  = gee_utils.get_terrain_mask()
        spectral_mask = gee_utils.make_mikan_mask(summer_ndvi, winter_ndvi, summer_ndmi)
        mask = spectral_mask.And(terrain_mask)
        area_m2 = mask.rename("mikan").multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=30, bestEffort=True,
        ).getInfo().get("mikan", 0)
        area_ha = round(area_m2 / 10000, 1)

        return {
            "year":            year,
            "ndvi_summer_mean": round(s_mean, 4),
            "ndvi_summer_std":  round(s_std, 4),
            "ndvi_winter":      round(w_mean, 4),
            "ndmi_summer":      round(ndmi_mean, 4),
            "area_sat_ha":      area_ha,
            "n_images":         n_images,
        }

    except Exception as e:
        print(f"  {year}年 取得失敗: {e}")
        return None


def fetch_landsat_feature_timeseries(
    start_year: int | None = None,
    end_year:   int | None = None,
) -> list[dict]:
    """年別Landsat特徴量を一括取得し、前年比面積変化（area_delta_ha）を追加する。"""
    start_year = start_year or CONFIG["landsat"]["start_year"]
    end_year   = end_year   or CONFIG["landsat"]["end_year"]

    records: list[dict] = []
    for year in range(start_year, end_year + 1):
        print(f"処理中: {year}年", end=" ")
        row = compute_annual_landsat_features(year)
        if row:
            # 前年比面積変化を追加
            if records:
                row["area_delta_ha"] = round(row["area_sat_ha"] - records[-1]["area_sat_ha"], 1)
            else:
                row["area_delta_ha"] = 0.0
            records.append(row)
            print(f"→ NDVI={row['ndvi_summer_mean']:.3f}, area={row['area_sat_ha']}ha")
        time.sleep(0.3)

    return records


def save_features_csv(records: list[dict], out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = PROCESSED_DIR / "features_landsat.csv"
    df = pd.DataFrame(records).sort_values("year")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"保存: {out_path}  ({len(df)}件)")
    return out_path


if __name__ == "__main__":
    gee_utils.init()
    print("=== Landsat年次特徴量取得 ===")
    records = fetch_landsat_feature_timeseries()
    save_features_csv(records)
