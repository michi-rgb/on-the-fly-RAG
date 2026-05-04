# On-the-fly RAG System

PDFファイルをアップロードして、Ollama（qwen3.5:latest）を使ってリアルタイムで情報を抽出・回答するWeb UIアプリケーションです。

## 📋 機能

- **PDFアップロード**: ドラッグ&ドロップでPDFを読み込み
- **カンマ区切り項目対応**: 質問にカンマで区切られた複数項目を指定し、各項目毎に検索・回答を生成
- **チャンク長さ調整**: GUI上でチャンク長さ（トークン数）をカスタマイズ可能
- **リアルタイム検索**: Ollama REST APIを使用した高速な埋め込みと検索
- **並列埋め込み処理**: 複数スレッドでチャンクを並列埋め込み（高速化）
- **自動リトライ**: Ollama接続エラー時に自動リトライ（指数バックオフ）
- **項目毎の引用**: 各項目の原文とページ番号を一緒に表示
- **グローバルチャンク化**: ページ境界を超えたチャンク分割で文脈を保持
- **シンプルなUI**: 直感的なWeb インターフェース
- **セッションベース**: PDFはセッション中のみ保持（永続化なし）

## 🛠️ 前提条件

- Python 3.8以上
- Ollama（localhost:11434で実行）

## 📦 セットアップ

### 1. Ollama のインストール・実行

https://ollama.ai から Ollama をダウンロード・インストール

```bash
# Ollama を起動
ollama serve

# 別のターミナルで qwen3.5:latest と nomic-embed-text-v2-moe モデルを pull
ollama pull qwen3.5:latest
ollama pull nomic-embed-text-v2-moe
```

### 2. Python 依存ライブラリのインストール

```bash
cd c:\Users\user\Documents\python_local\on-the-fly-RAG
pip install -r requirements.txt
```

## 🚀 実行方法

プロジェクトルートで以下を実行するだけです。

```powershell
python backend/main.py
```

起動すると以下が自動で行われます：

1. **FastAPI サーバー起動** — uvicorn が `http://localhost:8000` でリッスン開始
2. **Ollama 接続確認** — 起動時に `localhost:11434` への接続と埋め込みモデルの存在を確認（失敗した場合はエラーで停止）
3. **ブラウザ起動** — 2秒後にデフォルトブラウザで `http://localhost:8000` を自動オープン

停止するには `Ctrl+C` を押してください。

## 📁 プロジェクト構造

```
on-the-fly-RAG/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI アプリケーション
│   ├── pdf_processor.py     # PDF抽出・グローバルチャンク化
│   ├── ollama_client.py     # Ollama REST API クライアント（リトライ・接続再利用）
│   └── rag_engine.py        # RAG処理エンジン（カンマ区切り項目抽出・並列検索・生成）
├── frontend/
│   └── index.html           # Web UI（項目毎の結果表示、チャンク長さ調整対応）
├── requirements.txt         # Python 依存ライブラリ
└── README.md                # このファイル
```

## ⚙️ 設定

### チャンク分割設定
`backend/pdf_processor.py` の `process_pdf()` 関数で調整可能：
- `max_tokens_per_chunk`: デフォルト 100トークン（文字数÷4で計算）
- GUI上でアップロード時にカスタマイズ可能（50～5000トークン）
- **チャンク間のオーバーラップなし**: 各チャンクは重複なく分割されるため、チャンク境界をまたぐ情報は検索されにくい場合がある

### RAG検索設定
`backend/rag_engine.py` の `retrieve()` 関数で調整可能：
- `top_k`: 取得するチャンク数（デフォルト: 3）
- GUI上でクエリ時にカスタマイズ可能（1～10チャンク）

### 並列処理設定
`backend/rag_engine.py` の `RAGEngine` クラス：
- `embed_workers`: 並列埋め込みワーカー数（デフォルト: 4）
- 高速化のためHTTP接続を再利用（`requests.Session`）

### Ollama 設定
`backend/ollama_client.py` の `OllamaClient` クラスで調整可能：
- `base_url`: Ollama REST API URL（デフォルト: http://localhost:11434）
- `embedding_model`: 埋め込みモデル（デフォルト: nomic-embed-text-v2-moe）
- `generation_model`: 生成モデル（デフォルト: ）
- `retries`: エラー時のリトライ回数（デフォルト: 5回、指数バックオフ）
- `retry_backoff`: 初回バックオフ時間（デフォルト: 0.5秒）

アプリ起動時に Ollama へ接続し、`nomic-embed-text-v2-moe` がダウンロード済みか確認します。
未ダウンロードの場合は起動時にエラーとなり、`ollama pull nomic-embed-text-v2-moe` の実行を促します。

## 🔍 トラブルシューティング

### Ollamaに接続できない
```
✗ Ollamaに接続できません (localhost:11434)
```

**解決策:**
```bash
# Ollama が起動しているか確認
ollama serve

# 別ターミナルでモデルをpull
ollama pull llama3.1
ollama pull nomic-embed-text-v2-moe
```

### メモリ不足
llama3.1 や nomic-embed-text-v2-moe は大きなモデルです。メモリが限定的な場合、より小さいモデルの使用を検討してください：

```bash
ollama pull mistral
ollama pull tinydolphin
```

その場合、`backend/ollama_client.py` と `backend/main.py` でモデル名を変更してください。

### PDF処理エラー
アップロードしたPDFが破損していないか、テキスト抽出可能な形式か確認してください。

## 📝 使用方法の流れ

### 基本的な使い方

1. **Ollama起動**: `ollama serve` でバックエンドサーバー起動
2. **アプリ起動**: `run.bat` を実行（サーバー起動後、自動でブラウザが開く）
3. **PDFアップロード**: ドラッグ&ドロップでPDFを読み込み
4. **質問入力**: テキストボックスに質問を入力
5. **回答確認**: 原文とページ番号付きで結果が表示されます

### カンマ区切り項目の抽出例

質問文に**カンマ区切り**で項目を指定すると、各項目毎に検索・抽出を実行し、項目毎に回答と引用を表示します。

**質問例:**
```
製品名, 組成、成分情報, 適用法令, SDS改訂日
```

**動作:**
- 「製品名」に関連する情報を検索して回答 → 引用を表示
- 「組成、成分情報」に関連する情報を検索して回答 → 引用を表示
- 「適用法令」に関連する情報を検索して回答 → 引用を表示
- 「SDS改訂日」に関連する情報を検索して回答 → 引用を表示
