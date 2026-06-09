"""
prepare_dashboard_data.py
=========================
integrated_results.csv と market_data.csv を読み込み、
ダッシュボード用の JSON データを生成する。

感情スコアを [-1, 1] に正規化し、時間軸を分単位（0-60）に変換。
為替データは生の価格をそのまま出力する（ダッシュボード側で独立軸表示）。

出力: output/dashboard_data.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def normalize_to_range(series: pd.Series, target_min=-1.0, target_max=1.0) -> pd.Series:
    """Min-Max正規化 → [-1, 1]"""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return target_min + (series - s_min) / (s_max - s_min) * (target_max - target_min)


def main():
    integ_path = OUTPUT / "integrated_results.csv"
    market_path = OUTPUT / "market_data.csv"

    if not integ_path.exists():
        print(f"エラー: {integ_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(integ_path)
    print(f"integrated_results.csv: {df.shape}")

    # --- 時間軸を分に変換 ---
    t_min = df["start"].min()
    df["time_min"] = (df["start"] - t_min) / 60.0

    # --- 総裁セグメントのみ ---
    gov = df[df["is_governor"] == True].copy().sort_values("time_min").reset_index(drop=True)
    print(f"総裁セグメント: {len(gov)} 行")

    # --- 正規化 [-1, 1] ---
    gov["text_norm"] = normalize_to_range(gov["text_score"])
    gov["audio_norm"] = normalize_to_range(gov["audio_emotion_score"])
    gov["face_norm"] = normalize_to_range(gov["face_emotion_score"])

    # --- 乖離スコア（正規化後のスコアで再計算） ---
    gov["discrepancy_norm"] = (
        (gov["text_norm"] - gov["audio_norm"]).abs()
        + (gov["text_norm"] - gov["face_norm"]).abs()
    ) / 2.0  # 0-2 を 0-1 にスケール

    # 乖離フラグ（テキストがプラスかつ音声or表情がマイナス、またはその逆）
    gov["is_discrepancy"] = (
        ((gov["text_norm"] > 0.3) & ((gov["audio_norm"] < -0.2) | (gov["face_norm"] < -0.2)))
        | ((gov["text_norm"] < -0.3) & ((gov["audio_norm"] > 0.2) | (gov["face_norm"] > 0.2)))
    )

    # --- 為替データ（生価格をそのまま出力、正規化しない） ---
    market_data = []
    fx_min, fx_max = 0.0, 0.0
    if market_path.exists():
        mk = pd.read_csv(market_path)
        mk["Datetime"] = pd.to_datetime(mk["Datetime"])
        mk_min_time = mk["Datetime"].min()
        mk["time_min"] = (mk["Datetime"] - mk_min_time).dt.total_seconds() / 60.0
        fx_min, fx_max = float(mk["Close"].min()), float(mk["Close"].max())
        for _, row in mk.iterrows():
            market_data.append({
                "t": round(float(row["time_min"]), 2),
                "close": round(float(row["Close"]), 6),
            })
        print(f"市場データ: {len(market_data)} 行 (価格帯: {fx_min:.3f} - {fx_max:.3f})")
    else:
        print("market_data.csv なし → ダミー生成")
        for i in range(61):
            v = 140.5 + np.sin(i / 10.0) * 0.3 + np.random.randn() * 0.05
            market_data.append({
                "t": float(i),
                "close": round(v, 4),
            })
        fx_min, fx_max = 140.0, 141.0

    # --- JSON 出力 ---
    output_records = []
    for _, row in gov.iterrows():
        output_records.append({
            "t": round(float(row["time_min"]), 4),
            "text": round(float(row["text_norm"]), 4),
            "audio": round(float(row["audio_norm"]), 4),
            "face": round(float(row["face_norm"]), 4),
            "discrepancy": round(float(row["discrepancy_norm"]), 4),
            "is_disc": bool(row["is_discrepancy"]),
            "raw_text": round(float(row["text_score"]), 4),
            "raw_audio": round(float(row["audio_emotion_score"]), 4),
            "raw_face": round(float(row["face_emotion_score"]), 4),
            "utterance": str(row["text"])[:80],
            "speaker": str(row.get("speaker", "")),
        })

    data = {
        "segments": output_records,
        "market": market_data,
        "meta": {
            "total_segments": len(output_records),
            "duration_min": round(float(gov["time_min"].max()), 2),
            "generated_at": pd.Timestamp.now().isoformat(),
            "fx_min": round(fx_min, 4),
            "fx_max": round(fx_max, 4),
        }
    }

    out_path = OUTPUT / "dashboard_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    print(f"\n出力: {out_path}")
    print(f"セグメント数: {data['meta']['total_segments']}")
    print(f"会見時間: 0 - {data['meta']['duration_min']:.1f} 分")


if __name__ == "__main__":
    main()
