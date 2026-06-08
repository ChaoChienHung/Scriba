# Dev Notes

## 文件分工（Contract）

這個 repo 的文件分工採「單一事實來源」原則，避免內容重複與漂移：

- `AGENTS.md`：不可變更約束（Non-negotiables / Guardrails）
  - 放永遠要成立的規則：run 追溯規範、資料資產放置、安全/隱私、介面穩定性
  - 不放教學、評估設計、實驗流水帳
- `README.md`：對外入口（How to use）
  - 放快速上手、常用指令、啟動方式、路徑約定的使用方式
  - 不放長期研究計畫與實驗紀錄
- `docs/`：可演進但需要被固定成文字的規格與協作流程（roles/modules/workflows/api-compat/checklist）
- `docs/dev-notes.md`：設計決策 + 評估/benchmark + 實驗紀錄（Living doc）
  - 放「為什麼這樣設計」、評估/benchmark 規格、實驗模板/紀錄、坑與已知問題
  - 不複製貼上 `AGENTS.md` 的 non-regression 條款；需要時只引用 `AGENTS.md`
- `TODO.md`：Roadmap / Backlog（可勾選、可驗收）

小規則（寫文件時用來決策要放哪裡）：
- 任何「必須永遠成立」的句子 → 放 `AGENTS.md`
- 任何「第一次使用者要怎麼跑」 → 放 `README.md`
- 任何「為什麼這樣設計／這次實驗怎麼做、結果如何」 → 放 `docs/dev-notes.md`

## Evaluation Tiers

- Tier 0 (Smoke)
  - One batch forward + generate
  - Verify decoding works and no obvious NaNs
- Tier 1 (Offline metrics)
  - Run `python -m scriba.eval` on validation/test split
  - Track `cer` and `wer`
- Tier 2 (Qualitative)
  - Save a small set of predictions with image paths for manual inspection

## 評估資產與目錄（Evaluation Assets）

為了避免把「可長期比較/可版本控」的評估資產混進 `runs/`（一次性產物）或 `scriba/`（core package），固定測資與 benchmark 對照輸出集中在 `evaluation/`：

- `evaluation/eval_sets/`：固定離線測資
- `evaluation/results/`：對照輸出（model answers / judge）
- `evaluation/logs/`：跑評估命令的 stdout logs

## Training Observability（Logging / Metrics）

Paths and usage are documented in `README.md`. This section records the intent/semantics for experiment traceability.

- `runs/<run_name>/config.json`：run 參數與資料路徑快照（可重現）
- `trainer_state.json` / `log_history`：訓練過程的 step/epoch 指標時間序列（dashboard 用）
- `runs/<run_name>/eval/<split>/metrics.json`：離線 eval 的聚合指標（例如 cer/wer）
- `runs/<run_name>/eval/<split>/predictions.jsonl`：逐筆輸出（debug/誤差分析用）

## Web UI（FastAPI + Next.js）設計理念

### 動機

- Streamlit 足夠快，但對「長期要用的研究工具」而言，UI/UX 與資訊密度很難做到穩定、可擴展且好看
- 目標是做出一個可以一直長大的 OCR sandbox：可重現、可比較、可度量，而且能拿去 demo

### 核心哲學

- API-first：推論/模型管理/metrics 都由後端 API 提供，前端只負責呈現與互動
- Dashboard-first：重要資訊（source、latency、run、metric key）永遠可見，並且能快速切換
- Progressive disclosure：預設呈現「最關鍵」摘要；需要 debug 時才展開 raw JSON / raw logs
- Local-first：先把單機研究流程打順；多人共用/認證/併發屬於後續工程化（見 TODO）

### 資訊架構

- Inference：單模型推論（模型選擇 → decoding 參數 → 預覽 → 結果/latency/source）
- Comparison：2~4 模型對照（同一 input 橫向比對輸出與 latency）
- Training Metrics：run 選擇 → metric key 選擇 → 圖表 + 基本統計 + raw（debug）

### 一致性原則（為了研究效率）

- 模型來源抽象成同一套選項：latest / stored / pretrained / custom
- 每個推論結果至少包含：arch、source、latency、可序列化 output（方便做後處理/報表）
- UI 優先降低「切換成本」：不追求最花俏，而是讓比較/定位問題最快

### 延伸方向（先記錄動機）

- 多人/對外展示：需要併發控制、模型快取策略、資源隔離、日誌與觀測
- 認證：先 token、再 SSO/OAuth（目標是可控地共享 demo/工具）
