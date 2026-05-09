"""
筆ポリゴン前処理またはDynamic World代替サンプリング。

MAFFの筆ポリゴンポータル（open.fude.maff.go.jp）はログイン必須のため、
既にダウンロード済みGeoJSONがない場合はGoogle Dynamic Worldのcroplandラベルで
GEE内に訓練サンプルを生成する代替経路を提供する。
"""

from __future__ import annotations

import io
import json
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests
from shapely.geometry import shape

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


# ── 筆ポリゴン（MAFFポータル）──────────────────────────────────────────────

def try_download_fude(city_code_5: str, out_dir: Path) -> Optional[Path]:
    """
    MAFFポータル（open.fude.maff.go.jp）から筆ポリゴンZIPのダウンロードを試みる。
    ポータルはログイン必須のため通常403を返す。成功したPathを返し、失敗したらNoneを返す。
    手動ダウンロードの場合は data/raw/fude/ に geojson を配置し load_fude() を呼ぶこと。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # 注意: open.fude.maff.go.jp はユーザー登録・ログイン後のみダウンロード可能。
    # ここでは試みるが403が返った場合は None を返す。
    url = f"https://open.fude.maff.go.jp/datasets/{city_code_5}.zip"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 403:
            return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(out_dir)
        geojson_files = list(out_dir.glob("*.geojson"))
        return geojson_files[0] if geojson_files else None
    except Exception:
        return None


def load_fude(fude_dir: Path) -> gpd.GeoDataFrame:
    """解凍済みGeoJSONを読み込む。複数ファイルがある場合は結合する。"""
    files = list(fude_dir.glob("*.geojson"))
    if not files:
        raise FileNotFoundError(f"GeoJSONが見つかりません: {fude_dir}")
    frames = [gpd.read_file(f) for f in files]
    gdf = gpd.pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    print(f"読み込み: {len(gdf):,}ポリゴン ({[f.name for f in files]})")
    return gdf


def filter_by_landtype(gdf: gpd.GeoDataFrame, config: dict) -> gpd.GeoDataFrame:
    """
    地目コードでフィルタリングし label 列を追加する。
    land_type=200（畑）: label=1（みかん候補）
    land_type=100（田）: label=0（ネガティブサンプル）
    筆ポリゴンには作物種別コードが存在しないため、畑=みかん候補として扱う前提。
    """
    hata_code = config["fude"]["land_type_hata"]
    ta_code   = config["fude"]["land_type_ta"]
    area_min  = config["fude"]["area_min_m2"]

    landtype_col = next(
        (c for c in gdf.columns if "land_type" in c.lower() or "地目" in c),
        None,
    )
    if landtype_col is None:
        print(f"警告: 地目コード列が見つかりません。列一覧: {list(gdf.columns)}")
        raise ValueError("地目コード列が見つかりません")

    mask_hata = gdf[landtype_col] == hata_code
    mask_ta   = gdf[landtype_col] == ta_code
    print(f"畑（land_type={hata_code}）: {mask_hata.sum():,}件")
    print(f"田（land_type={ta_code}）  : {mask_ta.sum():,}件")

    filtered = gdf[mask_hata | mask_ta].copy()
    filtered["label"]     = (filtered[landtype_col] == hata_code).astype(int)
    filtered["land_type"] = filtered[landtype_col]

    filtered_proj = filtered.to_crs("EPSG:6677")
    filtered["area_m2"] = filtered_proj.geometry.area
    filtered = filtered[filtered["area_m2"] >= area_min].copy()
    print(f"面積フィルタ後: {len(filtered):,}ポリゴン（≥{area_min}m²）")
    return filtered.reset_index(drop=True)


def apply_terrain_flag(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """SRTM地形マスクで terrain_ok フラグを一括付与する。"""
    try:
        import ee
        import gee_utils
        gee_utils.init()

        terrain_mask = gee_utils.get_terrain_mask()
        wgs84 = gdf.to_crs("EPSG:4326").reset_index(drop=True)

        # ペイロード制限(10MB)を避けるため500件ずつバッチ処理
        BATCH = 500
        idx_to_flag: dict[int, int] = {}
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
            for feat in result_info.get("features", []):
                props = feat.get("properties", {})
                idx = props.get("idx")
                val = next((v for k, v in props.items() if k != "idx"), None)
                idx_to_flag[idx] = int(val or 0) if val is not None else 0
            print(f"  地形フラグ: {min(start+BATCH, len(wgs84))}/{len(wgs84)}完了", end="\r")

        gdf = gdf.copy()
        gdf["terrain_ok"] = [idx_to_flag.get(i, 0) for i in range(len(gdf))]
        ok_count = int(gdf["terrain_ok"].sum())
        print(f"\n地形フラグ: {ok_count:,}/{len(gdf):,}が南向き10〜40°")
    except Exception as e:
        print(f"警告: 地形フラグをスキップ（{e}）")
        gdf = gdf.copy()
        gdf["terrain_ok"] = 0

    return gdf


# ── Dynamic World 代替サンプリング ─────────────────────────────────────────

def create_training_samples_from_gee(
    n_mikan: int = 500,
    n_non_mikan: int = 500,
    year: int = 2022,
) -> gpd.GeoDataFrame:
    """
    筆ポリゴンが入手できない場合の代替: GEEの常緑マスク（スペクトル+地形）から訓練サンプル生成。

    注意: Dynamic World の crops ラベルはみかん（常緑樹）を trees と分類するため不適切。
    代わりに gee_utils.make_mikan_mask()（夏冬NDVI差 + NDMI + 地形）をラベルとして使用する。

    みかん候補ピクセル (label=1):
        make_mikan_mask() AND 地形マスク (南向き 10-40°) を両方満たすピクセル
    非みかんピクセル (label=0):
        地形マスクを満たすが make_mikan_mask() を満たさないピクセル（杉・檜林候補）

    各ピクセルを 30×30m の正方形ポリゴンに変換してGeoDataFrameを返す。
    """
    import ee
    import gee_utils
    gee_utils.init()

    aoi = gee_utils.get_aoi()

    # 夏冬Landsatコレクション（quality check なしで取得）
    summer_col = (
        gee_utils._get_landsat_collection(year, cloud_cover=30)
        .map(gee_utils.add_ndvi)
        .map(gee_utils.add_ndmi)
    )
    winter_col = gee_utils._get_winter_collection(year)

    if summer_col.size().getInfo() == 0 or winter_col.size().getInfo() == 0:
        # フォールバック: 隣接年を試す
        for fallback_year in [year - 1, year + 1, year - 2]:
            summer_col = gee_utils._get_landsat_collection(fallback_year).map(gee_utils.add_ndvi).map(gee_utils.add_ndmi)
            winter_col = gee_utils._get_winter_collection(fallback_year)
            if summer_col.size().getInfo() > 0 and winter_col.size().getInfo() > 0:
                print(f"  {year}年の画像なし → {fallback_year}年を使用")
                year = fallback_year
                break

    summer_ndvi = summer_col.median().select("NDVI")
    summer_ndmi = summer_col.median().select("NDMI")
    winter_ndvi = winter_col.median().select("NDVI")

    spectral_mask = gee_utils.make_mikan_mask(summer_ndvi, winter_ndvi, summer_ndmi)
    terrain_mask  = gee_utils.get_terrain_mask()

    mikan_mask     = spectral_mask.And(terrain_mask)       # 正例: みかん候補
    non_mikan_mask = spectral_mask.Not().And(terrain_mask)  # 負例: 同一地形だが非みかん（杉・檜）

    def sample_to_records(mask: ee.Image, label: int, n: int, source: str) -> list[dict]:
        samples = mask.selfMask().sample(
            region=aoi,
            scale=30,
            numPixels=n * 2,
            seed=42 + label,
            geometries=True,
        ).limit(n)
        info = samples.getInfo()
        records = []
        for feat in info.get("features", []):
            geom = feat["geometry"]
            cx, cy = geom["coordinates"]
            dx = dy = 0.000135  # ~15m in degrees at ~33°N
            poly = {
                "type": "Polygon",
                "coordinates": [[[cx-dx, cy-dy], [cx+dx, cy-dy], [cx+dx, cy+dy], [cx-dx, cy+dy], [cx-dx, cy-dy]]],
            }
            records.append({
                "label": label,
                "terrain_ok": 1,
                "area_m2": 900.0,
                "source": source,
                "geometry": shape(poly),
            })
        return records

    print(f"みかん候補サンプリング中（Landsatスペクトルマスク + 地形, {year}年）...")
    mikan_records = sample_to_records(mikan_mask, 1, n_mikan, "landsat_spectral")
    print(f"  {len(mikan_records)}件取得")

    print(f"非みかんサンプリング中（地形マスク内・スペクトル不合格）...")
    non_records = sample_to_records(non_mikan_mask, 0, n_non_mikan, "landsat_spectral")
    print(f"  {len(non_records)}件取得")

    all_records = mikan_records + non_records
    gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
    print(f"合計: {len(gdf)}サンプル (mikan={len(mikan_records)}, non={len(non_records)})")
    return gdf


def save_filtered(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_cols = [c for c in ["label", "land_type", "terrain_ok", "area_m2", "source", "geometry"] if c in gdf.columns]
    gdf[save_cols].to_crs("EPSG:4326").to_file(out_path, driver="GeoJSON")
    print(f"保存: {out_path}  ({len(gdf):,}件)")


if __name__ == "__main__":
    config = load_config()
    base = Path(__file__).parent.parent
    fude_dir = base / "data" / "raw" / "fude"
    out_path = base / "data" / "processed" / "fude_filtered.geojson"
    city_code = config["project"]["city_code_5"]

    # 既存GeoJSONがあればそれを使う
    existing = list(fude_dir.glob("*.geojson"))
    if existing:
        print(f"既存ファイルを使用: {[f.name for f in existing]}")
        gdf = load_fude(fude_dir)
        gdf = filter_by_landtype(gdf, config)
        gdf = apply_terrain_flag(gdf)
    else:
        # MAFFポータルダウンロードを試みる
        print(f"筆ポリゴンダウンロード試行: city={city_code}")
        result = try_download_fude(city_code, fude_dir)
        if result:
            print(f"ダウンロード成功: {result}")
            gdf = load_fude(fude_dir)
            gdf = filter_by_landtype(gdf, config)
            gdf = apply_terrain_flag(gdf)
        else:
            print()
            print("=" * 60)
            print("筆ポリゴンの自動ダウンロード不可（ログイン必須）")
            print("手動ダウンロード: https://open.fude.maff.go.jp/datasets/38204")
            print("→ GeoJSONを data/raw/fude/ に配置して再実行")
            print()
            print("代替: Dynamic World croplandサンプルを使用します...")
            print("=" * 60)
            gdf = create_training_samples_from_gee(n_mikan=500, n_non_mikan=500)

    save_filtered(gdf, out_path)
