# AGENTS

這份文件定義 Scriba repo 的不可變更約束（Non-negotiables / Guardrails）。任何會影響這些約束的改動，都必須先更新本文件並保持向後一致。

## 文件分工（單一事實來源）

- `AGENTS.md`：只放「必須永遠成立」的規則（本文件）
- `README.md`：使用入口（How-to / Quickstart）
- `docs/`：可演進但需要被固定成文字的規格與協作流程（例如：角色/模塊/流程/API 相容）
- `docs/dev-notes.md`：設計決策、評估/benchmark 規格、實驗紀錄（Living doc）
- `TODO.md`：Roadmap / Backlog（可勾選、可驗收）

## Doc Map

本文件是 `/docs/` 的索引入口：只做「導航 + 責任範圍」宣告，不複製各文件內容，避免規格漂移。

## Repo Core Docs（根目錄）

- [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md)：不可退化約束（Non-negotiables / Guardrails）與協作底線
- [README.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/README.md)：使用入口（Quickstart / 常用命令）
- [TODO.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/TODO.md)：Roadmap / Backlog（可勾選、可驗收）

## Docs（/docs）

- [docs/doc-map.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/doc-map.md)：本索引（本文件）
- [docs/roles.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/roles.md)：角色定義與責任邊界（RACI）
- [docs/modules.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/modules.md)：模塊邊界、核心職責與關鍵 I/O contract
- [docs/workflows.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/workflows.md)：標準工作流（train/eval/infer/模型生命週期）
- [docs/api-compat.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/api-compat.md)：API schema 版本化與相容策略
- [docs/checklist.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/checklist.md)：交付/驗收 checklist（合併前自檢）
- [docs/dev-notes.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/dev-notes.md)：設計決策、評估/benchmark 規格、實驗紀錄（Living doc）

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

## 協作責任邊界（RACI 的不可退化最小集）

- `scriba/` 的 contract（train/eval/inference/checkpoints/registry）必須有明確 owner；所有變更需至少 1 位 Core ML Owner review
- `apps/api` 的 schema 或欄位語意變更必須遵守版本化/相容策略，且需 App Owner review（相容策略見 `docs/api-compat.md`）
- `evaluation/`（可版本控制的評估資產）必須由 Data & Evaluation Owner 維護；不得把資料集內容與個資放入版本庫
- 角色定義與更完整的 RACI 規範放在 `docs/roles.md`，本節只保留不可退化的最低要求

## 文件同步（避免規格漂移）

- 任何改動若影響：
  - `runs/<run_name>/` 必備產物結構、`models/<arch>/latest` 行為、或資料/安全規則 → 必須同步更新本文件
  - 使用方式/命令列參數/常用工作流 → 必須同步更新 `README.md` 與 `docs/workflows.md`
  - API schema 相容策略 → 必須同步更新 `docs/api-compat.md` 與消費端（web/client）
  - 設計決策或評估方法學（為什麼/怎麼比較）→ 必須同步更新 `docs/dev-notes.md`

## 專屬指令處理（更新文件）

當使用者提出「請協助更新項目的相關文檔」或類似需求時，必須遵守以下流程：

- 先盤點 repo 內所有文件狀態（root + `/docs/` + 其他子模塊內 README/規範文件）
- 依 [docs/doc-map.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/doc-map.md) 的文件屬性與關係，判定優先級並分批更新
- 不要求一次做完所有文件；先更新高優先級入口與契約文件，再逐步擴散
  - 入口：`README.md`
  - 不可退化約束：`AGENTS.md`
  - 工作流：`docs/workflows.md`
  - API 相容：`docs/api-compat.md`
  - 模塊邊界：`docs/modules.md`
  - 角色/協作：`docs/roles.md`
  - 交付驗收：`docs/checklist.md`

## 安全與隱私

- 不要提交任何資料集內容、模型權重、cache、log、或包含個資的樣本到版本庫
- 若需要分享樣本，必須使用去識別化且可公開的最小資料
