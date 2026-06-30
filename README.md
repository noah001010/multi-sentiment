# Multi-Sentiment Analysis Project: Directory & Scripts Guide

本ドキュメントは、日銀総裁会見動画から「テキスト」「音声」「表情」の3つの感情（モダリティ）を抽出し、それらの乖離を検出・定量化して為替（市場）への影響を分析するマルチモーダル感情分析システムのディレクトリ構成および各スクリプトの役割を網羅的に解説したガイド（README）です。

---

## 📂 ディレクトリ構成一覧

```text
multi-sentiment/
├── README.md                 # 本ガイド（プロジェクト全体の説明・ファイル解説）
├── SYSTEM_ARCHITECTURE.md    # システムアーキテクチャ・設計ドキュメント
├── requirements.txt         # 依存Pythonライブラリ一覧
├── setup.sh                  # 環境構築・セットアップ用スクリプト
├── main.py                   # フル分析パイプラインを実行するメインスクリプト
├── app.py                    # メインのStreamlit分析ダッシュボード
├── app_advanced.py           # 動画と感情乖離点が連動するStreamlitカスタムUIプロトタイプ
├── run_analysis.py           # 5分デモ動画用の一括実行簡易スクリプト
├── create_mock_data.py       # テスト・デモ用のモックデータ生成スクリプト
├── update_mock.py            # デモ用にモックデータを更新するスクリプト
├── verify_gpu.py             # GPU (CUDA) の動作検証用スクリプト
├── verify_pipeline.py        # 小規模ダミーデータによるデータ統合ロジックの検証スクリプト
│
├── .venv/                    # Python仮想環境（依存ライブラリの隔離）
│
├── data/                     # 分析対象データおよびキャッシュ・モックデータ
│   ├── boj_conference.mp4    # 本番用会見動画
│   ├── boj_conference.wav    # 本番用会見音声 (mp4から自動抽出されたもの)
│   ├── boj_5min.mp4          # デモ用の5分間会見動画
│   ├── boj_5min.wav          # デモ用の5分間会見音声
│   ├── DAT_ASCII_USDJPY_M1_2023.csv # USD/JPYの為替1分足データ (HistData.com形式)
│   ├── mock_emotions.csv     # テスト用のモック表情・音声感情データ
│   ├── mock_financial.csv    # テスト用のモック為替リターンデータ
│   └── mock_text_sentiment.csv # テスト用のモックテキスト感情データ
│
├── src/                      # プロジェクトのコアロジック・再利用可能モジュール群
│   ├── preprocessing/        # 前処理モジュール (音声認識・話者分離・顔抽出)
│   │   ├── asd_pipeline.py       # 発話中の話者の顔画像を動画から切り出す処理
│   │   └── whisper_aligner.py    # Whisperを使用した文字起こしと時間同期
│   ├── features/             # 各モダリティの特徴量抽出モジュール
│   │   ├── text_analysis.py      # BERT / ModernBERT によるテキスト感情分析
│   │   ├── facial_analysis.py    # Py-Feat による表情分析 (Action Unit抽出)
│   │   ├── audio_analysis.py     # OpenSMILE による音声感情・トーン（韻律）分析
│   │   ├── multimodal_sentiment_pipeline.py # 特徴量抽出の統合パイプライン (表情+音声)
│   │   └── end_to_end_sentiment_pipeline.py # 3モダリティ抽出＋回帰分析まで行う統合パイプライン
│   ├── analysis/             # データのマージ、統計分析、可視化モジュール
│   │   ├── integrator.py         # 各種特徴量を時間軸でアライメントして乖離点を検出するコアロジック
│   │   ├── regression_analysis.py# 金融データとのマージ、OLS回帰分析 (HAC標準誤差) の実行
│   │   └── visualization.py      # Streamlitで表示するチャートの描画・生成処理
│   └── tools/                # ユーティリティツール
│       └── fetch_market_data.py  # 為替ヒストリカルデータなどの取得・整形ツール
│
├── static/                   # 静的アセット (Streamlitの動画Serving用)
│   └── boj_conference.mp4    # ダッシュボード上でブラウザから再生・シークするための動画ファイル
│
├── test_output/              # 統合テスト・検証スクリプトの出力先
│   ├── integrated_results.csv # 統合テストで出力されたCSV
│   ├── transcription_mock.csv # テスト用の文字起こしダミーCSV
│   └── visual_mock.csv       # テスト用の表情ダミーCSV
│
└── output/                   # 本番パイプライン実行結果の出力先 (自動作成されます)
    ├── integrated_results.csv # テキスト・表情・音声・話者が統合された最終結果CSV
    ├── transcription_clean.csv # クリーニング済みの文字起こしCSV
    ├── text_features.csv     # テキスト感情分析の中間結果CSV
    ├── facial_features_clean.csv # 表情分析の中間結果CSV
    ├── audio_features.csv     # 音声分析の中間結果CSV
    └── (その他 rawデータ、チャンクごとの中間ファイル等)
```

---

## 🔍 ユーザーの個別疑問への回答

### 1. なぜ `.mp4` (動画) と `.wav` (音声) の拡張子両方のファイルがあるの？
* **結論**: **音声分析ライブラリ（OpenSMILE や Whisper）が音声波形ファイル（`.wav`）を直接必要とするため**です。
* **仕組み**: 動画ファイル（`.mp4`）には映像と音声が混在しています。これらをそのまま読み込むとデコード負荷が高く、音声ライブラリで直接扱えません。そのため、`main.py` やパイプラインスクリプトは、内部的に `ffmpeg` コマンドを呼び出して `.mp4` から音声データのみを抜き出し、適切なフォーマット（16kHz、モノラルなど）の `.wav` ファイルとして保存し、それを音声処理モジュールに渡しています。

### 2. `mock` から始まる3つのCSVファイルは何のためにあるの？
* **結論**: **本物の重い分析を行わずに、プログラムの動作テストや Streamlit ダッシュボードの画面表示を確認するための「テスト用ダミーデータ」**です。
* **目的**: 本番の分析パイプライン（動画からの顔検出や音声処理など）は、実行に数十分〜数時間かかり、GPUも必要になります。開発時や、ダッシュボードのUI表示をすぐ確認したい時にこの「モックデータ」を読み込ませることで、一瞬で動作検証やUIレイアウトの調整ができるようにしています。

### 3. `src/` ディレクトリは本来何を入れるもので、ここでは何が入っているの？
* **本来の `src` (Source) フォルダの役割**: 
  プロジェクトの「コアロジック」「再利用可能なライブラリ（モジュール）」をまとめておくフォルダです。ルート直下の実行スクリプト（`main.py` や `app.py`）が雑多になるのを防ぎ、処理ごとにコードを部品化して整理するために存在します。
* **このプロジェクトにおける役割**: 
  前述の「ディレクトリ構成一覧」に記載の通り、**前処理 (preprocessing)**、**特徴量抽出 (features)**、**統計・乖離点分析 (analysis)**、**外部データ取得 (tools)** という役割ごとに部品化された Python ファイル群が整理されて格納されています。

### 4. `static/` フォルダとそこに入っている `.mp4` の役割は？
* **本来の `static` フォルダの役割**:
  Webアプリケーション（StreamlitやFlaskなど）が、外部のブラウザから画像、CSS、動画などの静的ファイルを直接URLでアクセスできるようにするための公開用フォルダです。
* **なぜ `.mp4` があるのか**:
  Streamlit 上に設置されたカスタム動画プレイヤー（HTML/JavaScriptで記述）が、**「乖離ポイントをクリックした際に動画の該当時間へジャンプする」**というシーク機能を実行するために、ブラウザから動画ファイルを読み込める場所（`/app/static/boj_conference.mp4`）に動画を公開しておく必要があるためです。

### 5. `test_output/` フォルダの役割と中身は？
* **結論**: **データ統合ロジックなどが正しくバグなく動くかをチェックするためのテスト結果格納フォルダ**です。
* **役割**: `verify_pipeline.py` を実行すると、このフォルダにダミーの文字起こしCSVや表情CSVが生成され、それらを結合するテストが行われます。プログラムのバグを探す際や、ロジック変更時のデバッグにのみ使用します。

### 6. `SYSTEM_ARCHITECTURE.md` とは何？
* **結論**: **本システムの全体設計・技術仕様を記述した仕様書（システム構成図）**です。
* **内容**: 動画の入力から、どのようなSOTA（最先端）モデル（Whisper-large-v3, pyannote.audio 3.1, Py-Feat ResNet-50, OpenSMILE, BERT-Japanese v3）を通って特徴量が抽出され、どのように秒単位で結合されて「乖離」を判断しているのかという「設計コンセプトとデータ構造の同期ロジック」が Mermaid 図とともにまとめられています。

---

## 🛠️ 各種 Python スクリプトファイルの詳細解説

### 1. ルート直下にあるスクリプト（メイン実行・アプリ・テスト検証用）

* **`main.py`**:
  日銀会見動画に対するフル分析パイプラインを実行するメインスクリプト。Whisperでの文字起こし、pyannoteによる話者分離、Py-Featによる表情分析、OpenSMILEによる音声分析を一括で（キャッシュを効かせつつ）処理し、最終的に `MultimodalIntegrator` で統合して `output/integrated_results.csv` を出力します。
* **`app.py`**:
  Streamlitで構築されたメインの感情分析可視化ダッシュボード。動画や為替データを指定して分析パイプライン（`end_to_end_sentiment_pipeline.py`）を実行し、時系列での感情変化、動画シーク付きの乖離点確認、為替リターンとのOLS回帰分析結果などを可視化します。
* **`app_advanced.py`**:
  `app.py` に組み込まれている「動画のクリック時間ジャンプ機能（シーク連動機能）」の挙動検証・開発用に作られた、HTML/JavaScriptとStreamlitを連携させたアドバンスドなUIデモアプリです。
* **`run_analysis.py`**:
  デモ動画（`boj_5min.mp4`）と為替データを指定して、バックエンドの分析パイプラインを一発で実行するための簡易ショートカットスクリプト。
* **`create_mock_data.py` / `update_mock.py`**:
  動作確認に使用するダミーのCSVデータ（`mock_emotions.csv` 等）を `data/` フォルダに新規作成・更新するスクリプト。
* **`verify_gpu.py`**:
  現在実行している環境の PyTorch が GPU (CUDA) を認識できているか、CUDA上での計算が正常に行えるかを確認する動作環境テストスクリプト。
* **`verify_pipeline.py`**:
  データ統合のコアクラスである `MultimodalIntegrator` が、文字起こし、表情、音声データを想定通り秒単位でアライメント・マージできるかを擬似的にテストするロジック検証スクリプト。結果は `test_output/` に出力されます。

### 2. `scripts/` ディレクトリ配下にあるスクリプト（特定処理・デバッグ用）

* **`compute_integrated_scores.py`**:
  `main.py` などで出力した中間ファイル（`audio_features.csv`, `facial_features_clean.csv`）を再ロードし、各モダリティの正規化感情スコア（`text_score`, `audio_emotion_score`, `face_emotion_score`）および乖離スコア（`discrepancy_score`, `discrepancy_score_3`）を計算し、`integrated_results.csv` を更新する計算処理用スクリプト。
* **`debug_diarization.py`**:
  pyannote.audio 3.1 を使って、指定した音声ファイルの「話者分離（Diarization）」だけをテスト実行するデバッグ用スクリプト。話者分離がうまくいかない場合や、戻り値のオブジェクト構造を開発環境で確認するために使用します。
* **`prepare_dashboard_data.py`**:
  統合されたCSV結果（`integrated_results.csv`）と市場為替データ（`market_data.csv`）を読み込み、ダッシュボードUIの表示に適した形に時間軸や正規化範囲を調整したJSONデータ（`output/dashboard_data.json`）をエクスポートする前処理スクリプト。
* **`run_text_only.py`**:
  動画や音声の重い特徴量抽出をスキップし、すでに文字起こしされたテキストデータに対してテキスト感情分析モデル（Fin-BERTやローカルのModernBERT等）だけを単体で実行して `output/text_features.csv` を再作成するテキスト分析専用スクリプト。
* **`test_audio.py`**:
  音声感情分析ライブラリである `opensmile` が現在の環境で正しくインストールされ、音声ファイルから正常に特徴量を抽出できるかをテストするスクリプト。
* **`test_hf_token.py`**:
  環境変数や `.env` の Hugging Face トークン（`HF_TOKEN`）が正しく読み込めているか、およびそれを用いて `pyannote/speaker-diarization-3.1` モデルのロードが正常に認証を通るかを確認するテストスクリプト。
* **`visualize_governor_scores.py`**:
  統合された `integrated_results.csv` から総裁（Governor）の発言部分のみをフィルタリングし、3つの感情モダリティ（テキスト・音声・表情）および感情の乖離スコアの時系列推移を、PNG画像（静的グラフ）およびPlotly HTML（動的グラフ）として `output/` に書き出す可視化スクリプト。

---

## 🔄 段階的ステップ実行手順（SSH/Linux環境向け）

以下は、一括処理ではなく、一つ一つのステップの実行結果を確認しながら進めるための手順です。

### 0. 準備: SSH環境（Linuxサーバー）でのコード取得と環境構築
Mac側でコードを作成・更新した後は、GitHubにPushしておきます。
その後、SSHでLinuxサーバー（`vector`）に接続し、以下のコマンドで最新コードを取得・環境準備します。
```bash
# SSHで接続しているターミナルにて
cd /path/to/multi-sentiment
git pull origin main

# 必要ライブラリのインストール（初回のみ）
pip install -r requirements.txt

# ★ 教授モデルの配置（初回のみ）
# モデルファイルはGitHub管理外のため、サーバーに直接配置する必要があります。
# text_model/model_32/checkpoint-25137/ に以下のファイルを配置:
#   - model.safetensors  (モデル重み)
#   - optimizer.pt       (オプティマイザー状態)
#   - training_args.bin  (訓練設定)
#   - rng_state.pth      (乱数状態)
#   - tokenizer.model    (トークナイザーモデル)
# ※ config.json, tokenizer.json, tokenizer_config.json などはGitに含まれます
```

### 1. Step 1: 音声認識 & 文字起こし (ASR)
動画から音声を抽出し、Whisperを用いて文字起こしCSV（タイムスタンプ付き）を作成します。
```bash
python scripts/run_step1_asr.py --video_path data/boj_5min.mp4 --output_path output/transcription.csv
```
*   **結果確認**: `output/transcription.csv` が作成され、テキストと開始・終了秒数が記述されていることを確認します。
*   **作成される副生成物**: 音声分析用に動画と同じ名前の `data/boj_5min.wav` が生成されます。

### 2. Step 2: 話者特定 (Diarization)
発話したのが誰（総裁か記者か）を特定する情報を生成します。
```bash
# ⮳ 環境変数 HF_TOKEN が設定されている必要があります。未設定の場合はエラーで停止します。
python scripts/run_step2_diarization.py --video_path data/boj_5min.mp4 --output_path output/raw/diarization.csv
```
*   **結果確認**: `output/raw/diarization.csv` が作成され、時間帯ごとの話者IDが記述されていることを確認します。

### 3. Step 3: 発話区間の顔画像切り出し (Face Crop)
話者分離の結果に基づき、発話時間中に映っている顔画像を動画から自動的に検出して切り出します。
```bash
# 総裁区間（SPEAKER_00）のみ切り出す場合
python scripts/run_step3_facecrop.py --video_path data/boj_5min.mp4 --diarization_path output/raw/diarization.csv --output_dir output/faces --speaker SPEAKER_00
```
*   **結果確認**: `output/faces/` ディレクトリの中に、切り出された大量の顔画像（例: `face_000120.jpg`）が存在することを確認します。

### 4. Step 4: テキスト感情分析
文字起こしテキスト（Step 1の出力）から、教授提供のModernBERT回帰モデルを用いて経済インパクトスコアを算出します。
```bash
python scripts/run_step4_text.py --input_path output/transcription.csv --output_path output/text_features.csv
```
*   **結果確認**: `output/text_features.csv` が作成され、各行に経済インパクトスコア（`sentiment_score`）が付与されていることを確認します。

### 5. Step 5: 表情感情分析 (Action Unit)
Step 3 で切り出された顔画像群から、Py-Feat を用いて表情のAction Unit特徴（AU04, AU12など）を抽出します。
```bash
python scripts/run_step5_facial.py --crop_dir output/faces --output_path output/facial_features_clean.csv
```
*   **結果確認**: `output/facial_features_clean.csv` にフレームごとの `AU04`, `AU12`, まばたき（`is_blink`）などの情報が記録されていることを確認します。

### 6. Step 6: 音声特徴・プロソディ抽出
文字起こしの各セグメント時間に合わせ、音声ファイル（`.wav`）からジッターや音量（Loudness）などのプロソディ特徴量を OpenSMILE を用いて抽出します。
```bash
python scripts/run_step6_audio.py --wav_path data/boj_5min.wav --transcription_path output/transcription.csv --output_path output/audio_features.csv
```
*   **結果確認**: `output/audio_features.csv` にセグメントごとの音響特徴量が記録されていることを確認します。

### 7. Step 7: マルチモーダルデータ統合 & 乖離度計算
Step 2, 4, 5, 6 のすべての中間特徴量CSVを読み込み、タイムスタンプをキーにしてマージし、感情の「乖離スコア (discrepancy_score)」を算出します。
```bash
# 分析対象の総裁話者ID（例: SPEAKER_00）を指定して統合
python scripts/run_step7_integration.py \
  --text_path output/text_features.csv \
  --facial_path output/facial_features_clean.csv \
  --audio_path output/audio_features.csv \
  --diarization_path output/raw/diarization.csv \
  --output_path output/integrated_results.csv \
  --governor_id SPEAKER_00
```
*   **結果確認**: `output/integrated_results.csv` にすべてのモダリティの感情スコアと、`discrepancy_score` （乖離スコア）などの追加列が作成されていることを確認します。

### 8. Step 8: 金融為替データマージ & OLS回帰分析
統合した感情スコアと、為替データ（CSV）をアライメントし、感情スコアが市場為替リターンに与える影響のOLS回帰分析（Newey-West HAC標準誤差）を実行します。
```bash
python scripts/run_step8_regression.py \
  --integrated_path output/integrated_results.csv \
  --financial_path data/DAT_ASCII_USDJPY_M1_2023.csv \
  --start_time "2023-06-16 15:30:00"
```
*   **結果確認**: ターミナルに回帰係数、p値、R-squared（決定係数）、VIF（多重共線性）の分析サマリー表が出力されることを確認します。
