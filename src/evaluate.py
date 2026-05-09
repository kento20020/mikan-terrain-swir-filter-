"""精度評価・可視化: Stage-1分類・Stage-2回帰の結果をレポートする。"""

from __future__ import annotations

import pickle
import tomllib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["axes.unicode_minus"] = False
# 日本語フォントが利用可能な場合のみ設定
for _font in ["IPAexGothic", "Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

BASE = Path(__file__).parent.parent
FIGURES_DIR = BASE / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Stage-1 評価 ─────────────────────────────────────────────────────────

def plot_feature_importance(pkl_path: Path, top_n: int = 20) -> Path:
    """RF特徴量重要度の棒グラフを出力する。"""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    rf = obj["model"]
    feature_cols = obj["feature_cols"]

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    importances[::-1].plot.barh(ax=ax)
    ax.set_title(f"Stage-1 RF Feature Importance TOP{top_n}")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    out_path = FIGURES_DIR / "stage1_feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"保存: {out_path}")
    return out_path


# ── Stage-2 評価 ─────────────────────────────────────────────────────────

def plot_stage2_timeseries(
    years: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    is_proxy: np.ndarray,
) -> Path:
    """
    実測値vs予測値の時系列グラフを生成する。
    疑似ラベル年（2007-2016）はハッチングで区別し、実測値ではないことを明示する。
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(years, actual / 1000, "o-", color="steelblue", label="Actual (direct data)", zorder=3)
    ax.plot(years, predicted / 1000, "s--", color="tomato", label="Predicted (LOOCV)", zorder=3)

    # 疑似ラベル年を網掛け
    proxy_years = years[is_proxy]
    if len(proxy_years) > 0:
        ax.axvspan(proxy_years.min() - 0.5, proxy_years.max() + 0.5,
                   alpha=0.08, color="gray", label="Proxy years (pref x 31%)")
        ax.plot(years[is_proxy], actual[is_proxy] / 1000,
                "o", color="gray", alpha=0.5, zorder=2)

    mape_direct = float(np.mean(np.abs((actual[~is_proxy] - predicted[~is_proxy]) / actual[~is_proxy])) * 100)
    ax.set_title(f"Yawatahama Mikan Shipment Prediction  MAPE (direct years): {mape_direct:.1f}%")
    ax.set_xlabel("Year")
    ax.set_ylabel("Shipment volume (1000 t)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "stage2_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"保存: {out_path}")
    return out_path


def print_summary_report(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print("Stage-2 評価サマリー")
    print("=" * 50)
    print(f"MAPE（直接測定年）: {metrics['mape_direct']:.1f}%  ← 主要指標")
    print(f"MAPE（疑似ラベル年）: {metrics['mape_proxy']:.1f}%  ※愛媛県計×31%推計")
    print(f"MAPE（全年）: {metrics['mape_all']:.1f}%")
    print()

    df = pd.DataFrame({
        "year":      metrics["years"],
        "actual_t":  metrics["actual"].astype(int),
        "pred_t":    metrics["loocv_preds"].astype(int),
        "error_pct": ((metrics["loocv_preds"] - metrics["actual"]) / metrics["actual"] * 100).round(1),
        "is_proxy":  metrics["is_proxy"],
    })
    print(df.to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE / "src"))
    from model import (
        BASE as MODEL_BASE,
        MODELS_DIR,
        load_stage1_data,
        load_stage2_data,
        train_stage1_rf,
        train_stage2_xgb,
        estimate_mikan_area,
    )

    feat_s2  = BASE / "data" / "processed" / "features_s2.geojson"
    feat_lc  = BASE / "data" / "processed" / "features_landsat.csv"
    labels   = BASE / "data" / "raw" / "stats" / "yawatahama_shipment.csv"
    rf_pkl   = MODELS_DIR / "stage1_rf.pkl"

    if rf_pkl.exists():
        plot_feature_importance(rf_pkl)

    if feat_lc.exists() and labels.exists():
        df = load_stage2_data(feat_lc, labels)
        area_ha = None
        if feat_s2.exists() and rf_pkl.exists():
            with open(rf_pkl, "rb") as f:
                obj = pickle.load(f)
            area_ha = estimate_mikan_area(obj["model"], feat_s2)

        _, metrics = train_stage2_xgb(df, stage1_area_ha=area_ha)
        print_summary_report(metrics)
        plot_stage2_timeseries(
            metrics["years"],
            metrics["actual"],
            metrics["loocv_preds"],
            metrics["is_proxy"],
        )
