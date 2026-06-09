"""
visualize_governor_scores.py
=============================
integrated_results.csv から総裁セグメントのみ抽出し、
3モダリティスコアと discrepancy_score の時系列を可視化する。

出力:
  output/governor_scores_timeseries.png  （matplotlib / 静的）
  output/governor_scores_timeseries.html （plotly / インタラクティブ）

使い方:
  .venv/bin/python scripts/visualize_governor_scores.py [--input output/integrated_results.csv]
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"


def seconds_to_mmss(sec: float) -> str:
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m:02d}:{s:02d}"


def plot_png(df: pd.DataFrame, out_path: Path) -> None:
    """4パネルの時系列グラフを PNG で保存。"""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Bank of Japan Governor — Multimodal Emotion Scores (Time Series)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(4, 1, hspace=0.55, figure=fig)

    series_config = [
        ("text_score",          "Text Score (ModernBERT)",       "#2196F3", True),
        ("audio_emotion_score", "Audio Emotion Score",            "#4CAF50", True),
        ("face_emotion_score",  "Face Emotion Score (AU12−AU04)", "#FF9800", True),
        ("discrepancy_score",   "Discrepancy Score",              "#F44336", False),
    ]

    x = df["start"]
    x_labels_pos = x[::max(1, len(x) // 8)]
    x_labels = [seconds_to_mmss(v) for v in x_labels_pos]

    for i, (col, title, color, draw_zero) in enumerate(series_config):
        ax = fig.add_subplot(gs[i])
        ax.plot(x, df[col], color=color, linewidth=1.2, alpha=0.8)
        ax.fill_between(x, 0, df[col], alpha=0.15, color=color)
        if draw_zero:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        if i == 3:
            ax.set_xlabel("Time (mm:ss)", fontsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PNG 保存: {out_path}")


def plot_html(df: pd.DataFrame, out_path: Path) -> None:
    """plotly でインタラクティブな HTML を生成。"""
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly が見つかりません。HTML 出力をスキップします。")
        return

    x_labels = [seconds_to_mmss(v) for v in df["start"]]

    series_config = [
        ("text_score",          "Text Score (ModernBERT)",       "#2196F3"),
        ("audio_emotion_score", "Audio Emotion Score",            "#4CAF50"),
        ("face_emotion_score",  "Face Emotion Score (AU12−AU04)", "#FF9800"),
        ("discrepancy_score",   "Discrepancy Score",              "#F44336"),
    ]

    def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
        """#RRGGBB -> 'rgba(R, G, B, alpha)' に変換"""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=[cfg[1] for cfg in series_config],
        vertical_spacing=0.07,
    )

    for i, (col, title, color) in enumerate(series_config, start=1):
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df[col],
                mode="lines",
                name=title,
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, 0.15),
            ),
            row=i, col=1,
        )

    fig.update_layout(
        title="Bank of Japan Governor — Multimodal Emotion Scores",
        height=900,
        showlegend=True,
        template="plotly_white",
        font=dict(size=11),
    )
    fig.update_xaxes(title_text="Time (mm:ss)", row=4, col=1)

    fig.write_html(str(out_path))
    logger.info(f"HTML 保存: {out_path}")


def main(input_path: Path, governor_id: str) -> None:
    if not input_path.exists():
        logger.error(f"ファイルが見つかりません: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # is_governor 列がなければ speaker で判定
    if "is_governor" not in df.columns:
        logger.warning("is_governor 列がありません。speaker で代替判定します。")
        df["is_governor"] = df.get("speaker", pd.Series()) == governor_id

    gov_df = df[df["is_governor"] == True].copy()
    logger.info(f"総裁セグメント: {len(gov_df)} 行 / 全 {len(df)} 行")

    if gov_df.empty:
        logger.error("総裁セグメントが 0 行です。governor_id を確認してください。")
        sys.exit(1)

    required = ["text_score", "audio_emotion_score", "face_emotion_score", "discrepancy_score"]
    missing = [c for c in required if c not in gov_df.columns]
    if missing:
        logger.error(
            f"必要な列が見つかりません: {missing}\n"
            "先に compute_integrated_scores.py を実行してください。"
        )
        sys.exit(1)

    gov_df = gov_df.sort_values("start").reset_index(drop=True)

    out_png = OUTPUT / "governor_scores_timeseries.png"
    out_html = OUTPUT / "governor_scores_timeseries.html"

    plot_png(gov_df, out_png)
    plot_html(gov_df, out_html)

    # discrepancy_score_3 も表示
    if "discrepancy_score_3" in gov_df.columns:
        logger.info(
            f"discrepancy_score_3: mean={gov_df['discrepancy_score_3'].mean():.4f}, "
            f"max={gov_df['discrepancy_score_3'].max():.4f}"
        )

    print("\n=== 総裁セグメント スコアサマリー ===")
    print(gov_df[required].describe().round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUT / "integrated_results.csv"),
        help="integrated_results.csv のパス",
    )
    parser.add_argument(
        "--governor_id",
        type=str,
        default="SPEAKER_15",
        help="総裁の speaker ID（is_governor 列がない場合に使用）",
    )
    args = parser.parse_args()
    main(Path(args.input), args.governor_id)
