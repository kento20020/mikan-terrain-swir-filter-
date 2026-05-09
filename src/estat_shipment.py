"""e-Stat APIから八幡浜市みかん出荷量を取得する。2007年以降は愛媛県計×推計シェアで補完。"""

from __future__ import annotations

import os
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY: str = os.environ["ESTAT_API_KEY"]
BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

RAW_STATS_DIR = Path(__file__).parent.parent / "data" / "raw" / "stats"
RAW_STATS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _get(endpoint: str, params: dict) -> dict:
    params["appId"] = API_KEY
    for _ in range(3):
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("e-Stat API: 3回リトライ後も失敗")


def fetch_direct_1993_2005() -> list[dict]:
    """
    果樹累年統計（0003274240）から八幡浜市のみかん出荷量（1993-2005）を取得する。
    cdArea=38204, cdCat01=120（出荷量）
    """
    records = []
    data = _get("getStatsData", {
        "statsDataId": CONFIG["estat"]["stats_id_cumulative"],
        "cdArea": "38204",
        "cdCat01": "120",  # みかん_出荷量
        "limit": 100,
    })
    try:
        values = data["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]
    except KeyError:
        print("  1993-2005年の市町村データが見つかりません（APIレスポンス不一致）")
        return records

    if isinstance(values, dict):
        values = [values]
    for v in values:
        year = int(str(v.get("@time", "0"))[:4])
        val = v.get("$", "-")
        if val not in ("-", "***", "X") and 1993 <= year <= 2005:
            records.append({"year": year, "shipment_t": float(val), "is_proxy": False})
    return records


def fetch_direct_2006() -> list[dict]:
    """
    市町村別統計（0003022267）から八幡浜市の2006年みかん出荷量を取得する。
    cat01=346（八幡浜市）, cat02=004（出荷量）
    """
    records = []
    data = _get("getStatsData", {
        "statsDataId": CONFIG["estat"]["stats_id_2006"],
        "cdCat01": "346",
        "cdCat02": "004",
        "limit": 10,
    })
    try:
        values = data["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]
    except KeyError:
        print("  2006年の市町村データが見つかりません（APIレスポンス不一致）")
        return records

    if isinstance(values, dict):
        values = [values]
    for v in values:
        val = v.get("$", "-")
        if val not in ("-", "***", "X"):
            records.append({"year": 2006, "shipment_t": float(val), "is_proxy": False})
    return records


def fetch_pref_shipment(start_year: int = 2007, end_year: int = 2016) -> pd.DataFrame:
    """
    愛媛県計みかん出荷量（statsId=0003313868, cat01=170/出荷量, cat02=0110）を取得する。
    疑似ラベル算出に使用する。
    """
    data = _get("getStatsData", {
        "statsDataId": CONFIG["estat"]["stats_id_pref"],
        "cdCat01": "170",
        "cdCat02": "0110",
        "cdArea": CONFIG["project"]["pref_code"],
        "limit": 50,
    })
    try:
        values = data["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]
    except KeyError:
        return pd.DataFrame()

    if isinstance(values, dict):
        values = [values]
    records = []
    for v in values:
        year = int(str(v.get("@time", "0"))[:4])
        val = v.get("$", "-")
        if val not in ("-", "***", "X") and start_year <= year <= end_year:
            records.append({"year": year, "pref_shipment_t": float(val)})
    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


def build_shipment_labels() -> pd.DataFrame:
    """
    直接取得（1993-2006）+ 愛媛県計×推計シェア（2007-2016）で出荷量ラベルを構築する。

    重要: 2007-2016年の出荷量は愛媛県計×31%（八幡浜市の推計シェア、2005-2006年実績平均）で推計。
    実測値ではない。誠実なMAPE評価は1993-2006年（直接測定年）のみで行うこと。
    """
    records: list[dict] = []

    print("1993-2005年データ取得中...")
    records.extend(fetch_direct_1993_2005())
    time.sleep(0.5)

    print("2006年データ取得中...")
    records.extend(fetch_direct_2006())
    time.sleep(0.5)

    print("2007-2016年疑似ラベル生成中...")
    pref_df = fetch_pref_shipment(2007, 2016)
    share = CONFIG["estat"]["yawatahama_share_proxy"]
    for _, row in pref_df.iterrows():
        records.append({
            "year": int(row["year"]),
            "shipment_t": round(row["pref_shipment_t"] * share, 1),
            "is_proxy": True,
        })

    df = (
        pd.DataFrame(records)
        .sort_values("year")
        .drop_duplicates("year")
        .reset_index(drop=True)
    )
    return df


def save_labels(df: pd.DataFrame) -> Path:
    path = RAW_STATS_DIR / "yawatahama_shipment.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    direct = df[~df["is_proxy"]]
    proxy  = df[df["is_proxy"]]
    print(f"保存: {path}")
    print(f"  直接測定年: {len(direct)}件 ({direct['year'].min()}-{direct['year'].max()})")
    print(f"  疑似ラベル: {len(proxy)}件  ({proxy['year'].min()}-{proxy['year'].max()} ※実測値ではない)")
    return path


if __name__ == "__main__":
    df = build_shipment_labels()
    print(df.to_string(index=False))
    save_labels(df)
