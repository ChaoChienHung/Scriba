# Checklist

本清單用於 feature/refactor 合併前後的自檢，避免破壞 core contract、可追溯性與應用層體驗。

## 變更類型 → 必做項

### A) 變更 `scriba/`（訓練/推論/評估 contract）

- `python3 -m scriba.train` 仍可跑通（至少 Tier 0：one batch forward + generate）
- `python3 -m scriba.eval` 仍可跑通（至少能產出 metrics）
- `runs/<run_name>/` 產物仍符合 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md) 的必備結構
- 若輸出格式/欄位有變更：
  - 更新 [docs/modules.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/modules.md)（I/O contract）
  - 更新 [apps/api](file:///Users/bytedance/Desktop/Ludwig/Scriba/apps/api) 與 [apps/web](file:///Users/bytedance/Desktop/Ludwig/Scriba/apps/web) 的消費端（若受影響）

### B) 變更評估（metrics/結果輸出/固定測資）

- `evaluation/` 的規範仍成立（見 [evaluation/README.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/evaluation/README.md)）
- 若新增固定測資或對照輸出：
  - 確認不含資料集內容、個資或未授權素材
  - `eval_sets/` 應僅存放可公開或去識別化的小型樣例

### C) 變更 `apps/api`（API schema）

- 遵守 [docs/api-compat.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/api-compat.md) 的相容規則
- Web UI 的 API client（`apps/web/src/lib/api.ts` 等）同步更新
- 新增欄位必須是 optional 或具 default，並維持舊 consumer 可用

### D) 變更 `apps/web`（UI 行為）

- Inference / Comparison / Metrics 三頁至少各做一次 smoke
- 預設模型來源（latest）仍可正常推論（若依賴 API/CLI）

### E) 更新 `models/<arch>/latest`

- `latest` 仍為 symlink 且指向存在的模型目錄
- CLI 推論不帶 `--model-dir` 仍能成功
- 若 `latest` 指向策略有變更，需更新 [docs/workflows.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/workflows.md) 與根目錄 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md)

## 文件同步（避免規格漂移）

- 改動「必須永遠成立」的規則 → 更新 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md)
- 改動「使用方式」→ 更新 [README.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/README.md) 與 [docs/workflows.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/workflows.md)
- 改動「設計決策/為什麼」→ 更新 [docs/dev-notes.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/dev-notes.md)
