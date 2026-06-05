# AGENTS

這份文件定義 Scriba repo 的不可變更約束（Non-negotiables / Guardrails）。任何會影響這些約束的改動，都必須先更新本文件並保持向後一致。

## 不退化原則

- `scriba/` 只放 core library（訓練/評估/核心推論/checkpoints contract）；可執行應用（API/Web/CLI）放 `apps/`
- `runs/<run_name>/` 是唯一實驗單位（單一 run 要能自我描述、可追溯、可重現）
- `models/<arch>/latest` 是「預設模型指標」且不得改成隱式的其他路徑

## 資料與產物管理

- `data/`、`runs/`、`models/` 皆視為本地資產，不得提交到 Git（僅允許 `.gitkeep` 或必要的小型範例檔）
- 不得在 repo 內寫入隱私資料、金鑰、token、帳密；不得在 log 中印出敏感資訊
- `models/<arch>/<name>/{model,processor}` 用於落地 pretrained / 可分享的 checkpoint；`latest` 以 symlink 指向某個模型目錄

## Train / Eval 入口（穩定介面）

- 訓練單一入口：`python3 -m scriba.train`
- 評估單一入口：`python3 -m scriba.eval`
- CLI 工具入口（應用層）：`python3 -m apps.cli.*`（例如 `apps.cli.infer`、`apps.cli.download`）

## Runs 追溯規範（必備產物）

- 每次訓練必須至少落地：
  - `runs/<run_name>/config.json`（run 設定快照）
  - `runs/<run_name>/model/`（HF `VisionEncoderDecoderModel`）
  - `runs/<run_name>/processor/`（對應 processor）
- 任何評估若指定 `--run-dir`，必須寫回：
  - `runs/<run_name>/eval/<split>/metrics.json`
  - （可選）`runs/<run_name>/eval/<split>/predictions.jsonl`

## 介面穩定性

- `apps/api` 的 API schema 與 `apps/web` 的 UI 行為可以演進，但必須維持清晰的版本化與相容策略（不要默默改掉回傳欄位語意）
- 任何「必須永遠成立」的規則只寫在這裡；其他文件要引用，不要複製貼上

## 安全與隱私

- 不要提交任何資料集內容、模型權重、cache、log、或包含個資的樣本到版本庫
- 若需要分享樣本，必須使用去識別化且可公開的最小資料

