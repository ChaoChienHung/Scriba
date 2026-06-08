# API Compatibility

本文件把 `apps/api` 的相容策略寫成可執行的規則，避免 API 回傳欄位語意被「默默改掉」。不可退化約束以 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md) 為準。

## 版本化策略

- 對外 API 必須有版本概念（至少先做到 v1）
  - 路徑版本：`/api/v1/...`（推薦）
  - 或 header 版本：`X-API-Version: 1`
- 同一個版本內必須相容（不破壞既有消費者）

## 相容規則（v1 內）

- 不可移除欄位；只能新增 optional 欄位
- 不可改變欄位語意（同名欄位不得改成不同含義）
- 不可把原本可能是 null 的欄位改成必填而無 default
- 任何會造成破壞性變更的需求，必須升版（v2），並保留 v1 一段時間

## 推論回傳（建議最小欄位）

- `arch`：模型架構（例如 trocr/donut）
- `source`：模型來源（latest/stored/pretrained/custom）
- `latency_ms`：推論延遲（用於比較與觀測）
- `output`：主要推論結果（文字或 structured JSON）
- `raw`（可選）：debug 用原始輸出或中間結果（需可關閉）

## 文件與驗收

- 任何 API schema 變更必須同步更新：
  - API schema（`apps/api/app/schemas.py` 或對應位置）
  - Web 端消費（若有）
  - [docs/checklist.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/checklist.md) 的驗收項
