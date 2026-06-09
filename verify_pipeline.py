import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# テスト用のモックデータ生成と統合ロジックの検証
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test-Verifier")

def create_mock_data(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("モックデータを生成中...")

    # 1. Mock Transcription (ASR)
    # 5つのセンテンス
    transcription_data = [
        {"start": 0.0, "end": 2.0, "text": "本日の金融政策決定会合において、"},
        {"start": 2.5, "end": 5.0, "text": "我々は長短金利操作の運用を柔軟化することを決定しました。"},
        {"start": 5.5, "end": 8.0, "text": "物価安定の目標実現が見通せる状況には至っていません。"},
        {"start": 8.5, "end": 10.0, "text": "粘り強く金融緩和を継続します。"},
        {"start": 10.5, "end": 12.0, "text": "リスクについては注視が必要です。"}
    ]
    pd.DataFrame(transcription_data).to_csv(output_dir / "transcription_mock.csv", index=False)
    
    # 2. Mock Visual Features (Py-Feat)
    # 30fps * 12秒 = 360フレーム
    frames = []
    for f in range(360):
        t = f / 30.0
        # 意図的に相関を作る: 5秒〜8秒（「見通せる状況には...」）の間、AU04（困惑）を高くする
        au04 = 0.8 if 5.5 <= t <= 8.0 else np.random.uniform(0, 0.3)
        au12 = 0.8 if 0.0 <= t <= 2.0 else np.random.uniform(0, 0.2)
        
        frames.append({
            "frame": f,
            "timestamp": t,
            "AU04": au04,
            "AU12": au12,
            "is_blink": 1 if np.random.random() < 0.05 else 0, # まばたき
            "EAR": 0.3
        })
    visual_df = pd.DataFrame(frames)
    visual_df.to_csv(output_dir / "visual_mock.csv", index=False)
    
    # 3. Mock Audio Features (OpenSMILE)
    audio_data = []
    for i in range(len(transcription_data)):
        audio_data.append({
            "sentence_id": i,
            "jitter": np.random.uniform(0.01, 0.05),
            "shimmer": np.random.uniform(0.1, 0.5),
            "F0_mean": 120.0 + np.random.normal(0, 10),
            "loudness": 0.5 + np.random.normal(0, 0.1)
        })
    audio_df = pd.DataFrame(audio_data)
    
    return transcription_data, visual_df, audio_df

def run_integration_test():
    output_dir = Path("test_output")
    trans_list, visual_df, audio_df = create_mock_data(output_dir)
    
    # Text Analysis Mock
    # センテンスごとの感情スコアを付与
    text_df = pd.DataFrame(trans_list)
    # "柔軟化"(ポジティブ), "至っていません"(ネガティブ), "緩和"(ポジティブ)
    text_df['sentiment_score'] = [0.1, 0.8, -0.6, 0.9, -0.2] 
    text_df['uncertainty_score'] = [0.0, 0.2, 0.8, 0.1, 0.9] # "注視"などで不確実性高
    
    # Audio結合
    text_audio = pd.concat([text_df, audio_df], axis=1)
    
    # Integration
    from src.analysis.integrator import MultimodalIntegrator
    integrator = MultimodalIntegrator()
    
    final_df = integrator.align_and_merge(text_audio, visual_df, audio_df)
    
    save_path = output_dir / "integrated_results.csv"
    final_df.to_csv(save_path, index=False)
    logger.info(f"統合テスト完了。結果を保存しました ({len(final_df)}行): {save_path}")
    logger.info("このCSVを使用してStreamlitの動作を確認できます。")
    
    # 検証：カラムの存在確認
    required_cols = ['text', 'sentiment_score', 'mean_AU04', 'jitter']
    for col in required_cols:
        if col not in final_df.columns:
            logger.error(f"必須カラム {col} が統合データに含まれていません！")
            sys.exit(1)
            
    logger.info("データ構造チェック: OK")

if __name__ == "__main__":
    run_integration_test()
