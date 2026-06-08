# Modules

本文件收斂各模塊的核心職責、關鍵輸入/輸出、以及與其他模塊的協同介面。不可退化約束仍以 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md) 為準。

## 分層總覽

- Core Library（`scriba/`）
  - 目的：訓練、評估、核心推論與 checkpoints contract。
  - 輸入：dataset（本地 `data/`）、模型設定（config/args）、checkpoint（本地 `models/` 或 `runs/`）。
  - 輸出：`runs/<run_name>/...` 產物、metrics、可被應用層消費的推論結果。
- Applications（`apps/`）
  - 目的：把 core 能力封裝成可使用的介面（CLI / API / Web）。
  - 輸入：core 提供的推論/模型載入能力、`models/<arch>/latest` 指標、`runs/` 與 `evaluation/` 資產。
  - 輸出：對外可操作的工具與 UI（推論、比較、metrics 視覺化）。
- Evaluation Assets（`evaluation/`）
  - 目的：收納「可版本控制、可長期累積」的評估與對照資產（見 [evaluation/README.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/evaluation/README.md)）。
  - 輸入：固定測資、模型對照輸出、judge 標註。
  - 輸出：可回歸、可比較的結果基準。

## 主要模塊

### `scriba/`（Core Library）

- `scriba.train`：訓練入口（`python3 -m scriba.train`）
  - 產物 contract：必須能產出 `runs/<run_name>/config.json`、`model/`、`processor/`（見 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md)）。
- `scriba.eval`：評估入口（`python3 -m scriba.eval`）
  - 若指定 `--run-dir`：必須回寫 `runs/<run_name>/eval/<split>/metrics.json`（可選 `predictions.jsonl`）。
- `scriba.inference`：核心推論能力（供 CLI/API/Web 呼叫）
  - 輸入：image、arch、checkpoint source（latest/stored/pretrained/custom）、decoding 參數。
  - 輸出：可序列化結果（文字或 structured JSON），並附上必要追溯資訊（arch/source/latency）。
- `scriba.models/*`：各 arch 的 config 與 model wrapper（例如 `trocr`、`donut`）
  - 約束：新增 arch 必須在 `scriba.models` 內自洽，並可被訓練/推論/評估共同消費。

### `apps/cli/`（CLI）

- `apps.cli.download`：下載 pretrained 到 `models/<arch>/<name>/{model,processor}`，並可更新 `models/<arch>/latest`
- `apps.cli.infer`：推論入口（單張圖/資料夾/批次；可輸出 JSONL）
- 約束：CLI 只做應用層封裝，核心邏輯優先下沉到 `scriba/` 以便 API/Web 共用。

### `apps/api/`（FastAPI）

- 目的：提供 inference、model 列表、run/metrics 查詢等 API 能力，供 Web UI 與其他工具消費。
- 介面穩定性：schema 可演進，但必須遵守版本化與相容策略（見 [api-compat.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/api-compat.md)）。

### `apps/web/`（Next.js）

- 目的：Research dashboard（Inference / Comparison / Training Metrics）。
- 原則：API-first、Dashboard-first、Progressive disclosure（詳見 [docs/dev-notes.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/dev-notes.md)）。

### `scripts/`（一次性腳本）

- 目的：提供可複用的下載/訓練模板，但不得變成核心依賴（核心入口仍以 `python3 -m scriba.*`、`python3 -m apps.cli.*` 為準）。
