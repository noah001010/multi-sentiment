import logging
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    Text sentiment/stance analyzer using a local ModernBERT regression model.

    教授提供の ModernBertForSequenceClassification (regression head) を使用。
    入力文を経済インパクトスコアとして連続値で返す。

    Score convention:
        score > 0  →  economically positive / hawkish
        score < 0  →  economically negative / dovish
        score ≈ 0  →  neutral
    """

    # デフォルトモデルパス（プロジェクトルートからの相対パス）
    DEFAULT_MODEL_DIR = "text_model/model_32"

    def __init__(
        self,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to a local HF-format model directory containing
                        a checkpoint-* subdirectory.
                        Defaults to text_model/model_32.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TextAnalyzer device: {self.device}")

        # モデルパスの解決
        if model_path is None:
            # プロジェクトルートからの相対パスを絶対パスに変換
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = str(project_root / self.DEFAULT_MODEL_DIR)

        self._load_local_model(model_path)

        # Uncertainty lexical markers (Hedges)
        self.hedge_words = [
            "と思われる", "考えられる", "可能性がある", "不透明", "注視する",
            "様子を見る", "リスク", "不確実性", "仮定", "シナリオ"
        ]

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _resolve_checkpoint_path(self, model_dir: str) -> Path:
        """
        モデルディレクトリから checkpoint-* サブディレクトリを自動検索する。
        教授の predict_score.py と同じロジック。

        1. model_dir 直下に config.json があればそのまま使用
        2. なければ checkpoint-* を検索
        """
        path = Path(model_dir)

        # 直下に config.json がある場合はそのまま使用
        if (path / "config.json").exists():
            return path

        # checkpoint-* サブディレクトリを検索
        checkpoints = sorted([
            d for d in path.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ])

        if not checkpoints:
            # さらに1段下を探索（model_32/checkpoint-25137 のような構造）
            for subdir in sorted(path.iterdir()):
                if subdir.is_dir():
                    sub_checkpoints = sorted([
                        d for d in subdir.iterdir()
                        if d.is_dir() and d.name.startswith("checkpoint-")
                    ])
                    if sub_checkpoints:
                        ckpt = sub_checkpoints[0]
                        logger.info(f"Found checkpoint: {ckpt}")
                        return ckpt

            raise FileNotFoundError(
                f"No checkpoint-* directory or config.json found in '{model_dir}'. "
                "Please provide a valid model directory."
            )

        ckpt = checkpoints[0]
        logger.info(f"Found checkpoint: {ckpt}")
        return ckpt

    def _load_local_model(self, model_dir: str) -> None:
        """Load a local HF model directory for inference."""
        ckpt_path = self._resolve_checkpoint_path(model_dir)
        logger.info(f"Loading model from: {ckpt_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path))
        self.model.to(self.device)
        self.model.eval()

        self.num_labels = self.model.config.num_labels
        self.problem_type = getattr(self.model.config, "problem_type", "")
        logger.info(
            f"Model loaded. num_labels={self.num_labels}, "
            f"problem_type={self.problem_type}, "
            f"id2label={getattr(self.model.config, 'id2label', 'N/A')}"
        )

    # ------------------------------------------------------------------
    # Public: single-sentence inference
    # ------------------------------------------------------------------

    def score_sentence(self, text: str) -> float:
        """
        Run inference on a single sentence and return a real-valued score.

        Returns:
            float: positive = economically positive/hawkish,
                   negative = economically negative/dovish.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, num_labels)

        logits = logits[0]  # (num_labels,)

        if (self.num_labels or 1) == 1 or self.problem_type == "regression":
            # Regression head: logit is already the score
            return float(logits[0].cpu())

        # Classification head: map probs to a single real score
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        id2label = getattr(self.model.config, "id2label", {})

        # Try to auto-detect label indices
        pos_idx, neg_idx = self._detect_label_indices(id2label, len(probs))
        return float(probs[pos_idx] - probs[neg_idx])

    def _detect_label_indices(self, id2label: dict, n: int):
        """Heuristically find positive / negative label indices."""
        pos_idx, neg_idx = 0, 1  # fallback
        for idx, label in id2label.items():
            label_l = str(label).lower()
            if any(k in label_l for k in ("positive", "pos", "hawkish", "up")):
                pos_idx = int(idx)
            elif any(k in label_l for k in ("negative", "neg", "dovish", "down")):
                neg_idx = int(idx)
        return pos_idx, neg_idx

    # ------------------------------------------------------------------
    # Public: batch / segment analysis (primary interface)
    # ------------------------------------------------------------------

    def analyze_texts(
        self,
        texts: List[str],
        starts: Optional[List[float]] = None,
        ends: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Analyze a list of sentences and return a DataFrame.

        When starts/ends are provided each row represents one sentence and
        includes all aggregation columns computed over that single sentence
        (sentence_count=1).  Pass them when the caller aggregates at a
        higher level.

        Aggregation columns always present:
            text_score_mean, text_score_sum,
            positive_sentence_ratio, negative_sentence_ratio,
            sentence_count

        Backward-compatible columns:
            sentiment_score  (= text_score_mean)
            sentiment_label  ('POSITIVE' / 'NEGATIVE' / 'NEUTRAL')
            uncertainty_score
        """
        logger.info(f"Analyzing {len(texts)} sentences...")

        rows = []
        for text in texts:
            score = self.score_sentence(text)
            hedge_count = sum(1 for w in self.hedge_words if w in text)
            uncertainty = min(1.0, hedge_count / max(1, len(text) / 20))

            rows.append({
                "text": text,
                "_raw_score": score,
                "uncertainty_score": uncertainty,
            })

        df = pd.DataFrame(rows)

        # Aggregation columns (sentence-level: each row is its own "segment")
        df["text_score_mean"] = df["_raw_score"]
        df["text_score_sum"] = df["_raw_score"]
        df["positive_sentence_ratio"] = (df["_raw_score"] > 0).astype(float)
        df["negative_sentence_ratio"] = (df["_raw_score"] < 0).astype(float)
        df["sentence_count"] = 1

        # Backward-compatible aliases
        df["sentiment_score"] = df["_raw_score"]
        df["sentiment_label"] = df["_raw_score"].apply(
            lambda s: "POSITIVE" if s > 0 else ("NEGATIVE" if s < 0 else "NEUTRAL")
        )

        df = df.drop(columns=["_raw_score"])

        if starts is not None:
            df["start"] = starts
        if ends is not None:
            df["end"] = ends

        return df

    def aggregate_by_segment(
        self,
        sentence_df: pd.DataFrame,
        segment_starts: List[float],
        segment_ends: List[float],
    ) -> pd.DataFrame:
        """
        Aggregate sentence-level scores into fixed time-segments.

        Args:
            sentence_df: Output of analyze_texts() with 'start' column.
            segment_starts: List of segment start times (seconds).
            segment_ends: List of segment end times (seconds).

        Returns:
            DataFrame indexed by segment, one row per segment.
        """
        records = []
        for seg_start, seg_end in zip(segment_starts, segment_ends):
            subset = sentence_df[
                (sentence_df["start"] >= seg_start) & (sentence_df["start"] < seg_end)
            ]
            if subset.empty:
                records.append({
                    "seg_start": seg_start, "seg_end": seg_end,
                    "text_score_mean": 0.0, "text_score_sum": 0.0,
                    "positive_sentence_ratio": 0.0, "negative_sentence_ratio": 0.0,
                    "sentence_count": 0,
                })
            else:
                scores = subset["sentiment_score"]
                records.append({
                    "seg_start": seg_start, "seg_end": seg_end,
                    "text_score_mean": float(scores.mean()),
                    "text_score_sum": float(scores.sum()),
                    "positive_sentence_ratio": float((scores > 0).mean()),
                    "negative_sentence_ratio": float((scores < 0).mean()),
                    "sentence_count": len(scores),
                })
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Self-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        default=None,
        help="Path to a local HF model directory (optional). "
             "Omit to use the default professor model (text_model/model_32).",
    )
    args = ap.parse_args()

    analyzer = TextAnalyzer(model_path=args.model_path)

    sample_texts = [
        "物価目標の実現が見通せる状況になった。",           # expected: positive
        "先行きについては極めて不確実性が高いと思われる。",   # expected: negative
        "賃金と物価の好循環が確認されつつあります。",         # expected: positive
    ]

    print("\n=== sentence-level scores ===")
    for t in sample_texts:
        s = analyzer.score_sentence(t)
        print(f"  [{s:+.4f}] {t}")

    print("\n=== analyze_texts() output ===")
    starts = [0.0, 5.0, 10.0]
    ends   = [5.0, 10.0, 15.0]
    df = analyzer.analyze_texts(sample_texts, starts=starts, ends=ends)
    print(df.to_string(index=False))

    # Validate required columns are present
    required_cols = {
        "text_score_mean", "text_score_sum",
        "positive_sentence_ratio", "negative_sentence_ratio",
        "sentence_count", "sentiment_score", "sentiment_label",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"\n❌ MISSING COLUMNS: {missing}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✅ All required columns present.")

    # score_sentence must return float
    assert isinstance(analyzer.score_sentence(sample_texts[0]), float), \
        "score_sentence must return float"
    print("✅ score_sentence() returns float.")

    print("\nAll tests passed.")
