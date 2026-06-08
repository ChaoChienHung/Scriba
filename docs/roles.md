# Roles

本文件定義 Scriba 在協作時的「責任邊界」與最小 RACI，目標是讓每個改動都能找到 owner、reviewer 與驗收點。不可退化約束仍以 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md) 為準。

## 角色

- Repo Maintainer（維護者）
  - 責任：Guardrails 的最終把關、repo 結構治理、合併規則、版本策略。
- Core ML Owner（核心訓練/推論）
  - 責任：`scriba/` 內訓練/評估/推論 contract、模型註冊與 arch 介面一致性、checkpoints contract。
- Data & Evaluation Owner（資料/評估）
  - 責任：`evaluation/` 的固定測資與長期對照資產；評估指標定義與評估流程；確保不把資料集內容提交到 Git。
- App Owner（應用層：API/Web/CLI）
  - 責任：`apps/api`、`apps/web`、`apps/cli` 的端到端體驗；API schema 相容策略與 UI 行為一致。
- Infra / Release Owner（工程化/發版）
  - 責任：依賴管理（requirements / Node deps）、CI、部署/打包流程、模型快取與資源限制策略（若有）。

## RACI（最小規則）

- 變更 `scriba/`（train/eval/inference/checkpoints/registry）
  - A：Core ML Owner
  - R：改動者
  - C：Data & Evaluation Owner（若影響 eval 指標/輸出）
  - I：App Owner（若影響 apps 消費的介面）
- 變更 `apps/api` schema 或欄位語意
  - A：App Owner
  - R：改動者
  - C：Core ML Owner（若影響推論輸出結構）、apps/web owner（若 UI 消費）
  - I：Repo Maintainer
- 變更 `apps/web` 的頁面行為（Inference/Comparison/Metrics）
  - A：App Owner
  - R：改動者
  - C：Core ML Owner（推論參數/輸出）、Data & Evaluation Owner（metrics 命名/來源）
  - I：Repo Maintainer
- 變更 `evaluation/`（eval_sets、model_answers、judge）
  - A：Data & Evaluation Owner
  - R：改動者
  - C：Core ML Owner（若改指標或輸出格式）
  - I：Repo Maintainer
- 更新 `models/<arch>/latest` 指向規則或工具行為
  - A：Core ML Owner
  - R：改動者
  - C：App Owner（CLI/Web/API 預設行為）
  - I：Repo Maintainer

## Review Gate（建議）

- 改動任何「契約」時，PR 需包含：
  - 一個可重現的命令（或腳本）可產生/驗證變更
  - 受影響文件更新（見 [docs/doc-map.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/doc-map.md) 的文件分工）
