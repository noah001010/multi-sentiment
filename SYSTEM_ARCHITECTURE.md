# システム構成図 (System Architecture) - SOTA Overhaul v3.1

本プロジェクトは、日銀総裁会見の動画からマルチモーダル信号を抽出し、発言内容と非言語情報の乖離を定量化することで、市場への潜在的影響を分析するSOTA（最先端）パイプラインです。

## 1. 全体フロー
パイプラインは「前処理」「特徴量抽出」「データ統合」「可視化」の4フェーズで構成されます。

```mermaid
graph TD
    A[動画入力: boj_conference.mp4] --> B{前処理}
    B --> B1[ASR: Whisper-large-v3]
    B --> B2[Diarization: pyannote-audio 3.1]
    B --> B3[Face Crop: RetinaFace]
    
    B1 --> C{特徴量抽出}
    B2 --> C
    B3 --> C
    
    C --> C1[Text: BERT-Japanese Sentiment]
    C --> C2[Facial: Py-Feat ResNet-50 AUs]
    C --> C3[Audio: OpenSMILE eGeMAPS]
    
    C1 & C2 & C3 --> D[データ統合: MultimodalIntegrator]
    D --> E[分析ダッシュボード: Streamlit]
    D --> F[生データログ: output/raw/]
```

## 2. モジュール仕様 (SOTA Components)

| カテゴリ | コンポーネント | 技術仕様 | 目的 |
| :--- | :--- | :--- | :--- |
| **音声文字起こし** | Whisper-large-v3 | float16 / GPU加速 | 極めて高い認識精度とタイムスタンプの同期 |
| **話者特定** | pyannote.audio 3.1 | Gated SOTA Model | 総裁と記者の発言区間をミリ秒単位で分離 |
| **表情分析** | Py-Feat (ResNet-50) | ResNet-50 AUモデル | 20種類以上のAction Unit (AU) 高精度抽出 |
| **音響分析** | OpenSMILE | eGeMAPSv02 セット | 声の緊張度、F0、ラウドネスの抽出 |
| **感情分析** | BERT-Japanese v3 | cl-tohoku/bert-base | 日本語文脈に特化したセンチメントスコア |

## 3. データ統合 & 乖離ロジック (Sync Logic)
- **Time Sync**: 全モダリティを秒単位のタイムスタンプでアライメント。
- **Speaker Alignment**: 話者分離結果に基づき、総裁区間のみを抽出。
- **Discrepancy Reasoning**: 
    - 文言（Sentiment）と表情（Action Unit）の負の相関を検知。
    - 例：「緩和継続」というポジティブな言葉に対し、AU04（眉間の寄せ）が強い場合、潜在的な警戒感としてスコア化。

## 4. 研究の透明性 (Accountability)
- 全ての抽出プロセスにおいて、フィルタリング前の生データを `output/raw/` に記録。
- OpenSMILEやPy-Featの抽出失敗（フラットライン）を自動検知する警告機能を実装。
