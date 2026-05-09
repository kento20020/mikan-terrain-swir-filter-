"""Sentinel-2/1 + SRTM地形の87バンド特徴量をポリゴン単位で生成する。"""

from __future__ import annotations

import sys
import time
import tomllib
from pathlib import Path

import geemap
import geopandas as gpd

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

# gee_utils を GEE/src から参照
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import ee
import gee_utils  # add_ndvi, add_ndmi, get_terrain_mask を再利用

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
EXPORTS_DIR   = Path(__file__).parent.parent / "data" / "raw" / "gee_exports"


def get_yawatahama_aoi() -> ee.Geometry:
    c = CONFIG["aoi"]["coords"]
    return ee.Geometry.Rectangle(c)


# ── バンドリネーム・インデックス追加 ────────────────────────────────────────

def rename_s2_bands(img: ee.Image) -> ee.Image:
    """S2バンドをgee_utils互換名（Red/NIR/SWIR1/RedEdge/Green）にリネームする。"""
    return img.select(
        ["B4", "B8", "B11", "B5", "B3"],
        ["Red", "NIR", "SWIR1", "RedEdge", "Green"],
    )


def add_ndre(img: ee.Image) -> ee.Image:
    """NDre = (NIR - RedEdge) / (NIR + RedEdge)。みかん着果期の葉緑素に敏感。"""
    ndre = img.normalizedDifference(["NIR", "RedEdge"]).rename("NDre")
    return img.addBands(ndre)


def add_gndvi(img: ee.Image) -> ee.Image:
    """GNDVI = (NIR - Green) / (NIR + Green)。葉面積指数と相関。"""
    gndvi = img.normalizedDifference(["NIR", "Green"]).rename("GNDVI")
    return img.addBands(gndvi)


def add_ndwi(img: ee.Image) -> ee.Image:
    """NDWI = (Green - NIR) / (Green + NIR)。水体・湿潤検出。"""
    ndwi = img.normalizedDifference(["Green", "NIR"]).rename("NDWI")
    return img.addBands(ndwi)


# ── 月次コンポジット ────────────────────────────────────────────────────────

def make_s2_monthly_composite(year: int, month: int, aoi: ee.Geometry) -> ee.Image:
    """Sentinel-2月次コンポジット（Cloud Score+マスク済み）を4指数バンドで返す。"""
    start = f"{year}-{month:02d}-01"
    end   = f"{year}-{month:02d}-28" if month != 12 else f"{year}-12-31"

    threshold = CONFIG["sentinel2"]["cloud_threshold"]

    s2_col = ee.ImageCollection(CONFIG["sentinel2"]["collection"])
    cs_col = ee.ImageCollection(CONFIG["sentinel2"]["cloud_score_collection"])

    # CS+コレクションをJoinしてcs_scoreバンドを追加（linkCollectionのnull問題を回避）
    join = ee.Join.saveFirst("cs_image")
    condition = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = ee.ImageCollection(
        join.apply(
            s2_col.filterDate(start, end).filterBounds(aoi),
            cs_col.filterDate(start, end).filterBounds(aoi),
            condition,
        )
    )

    def mask_and_scale(img: ee.Image) -> ee.Image:
        cs_img = ee.Image(img.get("cs_image")).select("cs")
        masked = img.updateMask(cs_img.gt(threshold)).multiply(0.0001)
        return masked

    composite = joined.map(mask_and_scale).median()
    renamed = rename_s2_bands(composite)

    indices = gee_utils.add_ndvi(renamed)
    indices = add_ndre(indices)
    indices = gee_utils.add_ndmi(indices)
    indices = add_ndwi(indices)
    indices = add_gndvi(indices)
    suffix = f"_{month:02d}"
    return indices.select(["NDVI", "NDre", "NDMI", "NDWI", "GNDVI"]).rename(
        [f"NDVI{suffix}", f"NDre{suffix}", f"NDMI{suffix}", f"NDWI{suffix}", f"GNDVI{suffix}"]
    )


def make_s1_monthly_composite(year: int, month: int, aoi: ee.Geometry) -> ee.Image:
    """Sentinel-1 IW月次平均コンポジットをVH/VV/ratioで返す。"""
    start = f"{year}-{month:02d}-01"
    end   = f"{year}-{month:02d}-28" if month != 12 else f"{year}-12-31"

    col = (
        ee.ImageCollection(CONFIG["sentinel1"]["collection"])
        .filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .mean()
    )
    ratio = col.select("VH").subtract(col.select("VV")).rename("VH_VV_ratio")
    composite = col.addBands(ratio)
    suffix = f"_{month:02d}"
    return composite.rename([f"VH{suffix}", f"VV{suffix}", f"VH_VV_ratio{suffix}"])


def get_terrain_features() -> ee.Image:
    """SRTM DEMから連続値の地形特徴量3バンド（elevation/slope/aspect）を返す。"""
    dem    = ee.Image(CONFIG["dem"]["collection"]).select(CONFIG["dem"]["band"])
    slope  = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    return ee.Image.cat([dem.rename("elevation"), slope, aspect])


# ── 87バンド特徴量イメージ ────────────────────────────────────────────────

def make_feature_image(year: int | None = None) -> ee.Image:
    """
    S2 (48バンド) + S1 (36バンド) + 地形 (3バンド) = 87バンドの特徴量イメージを生成する。
    S2とS1はmonthly compositeのmedian, 地形は静的画像。
    """
    if year is None:
        year = CONFIG["sentinel2"]["year"]

    aoi = get_yawatahama_aoi()
    bands: list[ee.Image] = []

    for month in range(1, 13):
        bands.append(make_s2_monthly_composite(year, month, aoi))
        bands.append(make_s1_monthly_composite(year, month, aoi))

    bands.append(get_terrain_features())
    return ee.Image.cat(bands)  # 4×12 + 3×12 + 3 = 87


# ── ポリゴン単位特徴量抽出 ───────────────────────────────────────────────

def _gdf_to_ee_fc(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """GeoDataFrame（WGS84）をee.FeatureCollectionに変換する。"""
    features = []
    for idx, row in gdf.iterrows():
        geom_json = row.geometry.__geo_interface__
        props = {"polygon_id": int(idx), "label": int(row.get("label", -1))}
        if "terrain_ok" in row:
            props["terrain_ok"] = int(row["terrain_ok"])
        if "area_m2" in row:
            props["area_m2"] = float(row["area_m2"])
        features.append(ee.Feature(ee.Geometry(geom_json), props))
    return ee.FeatureCollection(features)


def extract_polygon_features(
    fude_gdf: gpd.GeoDataFrame,
    year: int | None = None,
    batch_size: int = 500,
) -> gpd.GeoDataFrame:
    """
    87バンド特徴量をポリゴン単位（中央値）で抽出する。
    ee.data.computeFeatures() を使いDriveを使用しない。
    ポリゴン数が多い場合はbatch_size単位でページ処理する。
    """
    img = make_feature_image(year)
    scale = CONFIG["sentinel2"]["scale_m"]

    all_frames: list[gpd.GeoDataFrame] = []
    wgs84 = fude_gdf.to_crs("EPSG:4326")

    for start in range(0, len(wgs84), batch_size):
        batch = wgs84.iloc[start:start + batch_size]
        fc = _gdf_to_ee_fc(batch)
        reduced = img.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=scale,
            tileScale=4,
        )
        print(f"  抽出中: ポリゴン {start+1}-{min(start+batch_size, len(wgs84))}/{len(wgs84)}", end=" ")
        result = geemap.ee_to_gdf(reduced)
        all_frames.append(result)
        print(f"→ {len(result)}件")
        time.sleep(0.5)

    return gpd.pd.concat(all_frames, ignore_index=True) if all_frames else gpd.GeoDataFrame()


def export_to_drive(fude_gdf: gpd.GeoDataFrame, year: int | None = None, description: str = "mikan_features") -> None:
    """
    Drive経由でGeoJSONとしてエクスポートする（フォールバック: ポリゴン数5,000超時）。
    ジョブ完了まで30秒間隔でポーリングする。
    """
    img = make_feature_image(year)
    scale = CONFIG["sentinel2"]["scale_m"]
    fc = _gdf_to_ee_fc(fude_gdf.to_crs("EPSG:4326"))
    reduced = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale)

    task = ee.batch.Export.table.toDrive(
        collection=reduced,
        description=description,
        fileFormat="GeoJSON",
        folder="gee_exports_mikan",
    )
    task.start()
    print(f"GEE Exportジョブ開始: {description}")
    while task.status()["state"] in ("READY", "RUNNING"):
        print(f"  状態: {task.status()['state']}")
        time.sleep(30)
    print(f"完了: {task.status()['state']}")


if __name__ == "__main__":
    gee_utils.init()
    print("=== 87バンド確認 ===")
    img = make_feature_image(2022)
    bands = img.bandNames().getInfo()
    print(f"バンド数: {len(bands)}")
    print(f"先頭5: {bands[:5]}")
    print(f"末尾5: {bands[-5:]}")
    assert len(bands) == 99, f"期待99バンド、実際{len(bands)}"  # S2:5指数×12=60, S1:3×12=36, 地形:3
    print("OK")
