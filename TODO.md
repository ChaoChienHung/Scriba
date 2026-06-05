# TODO

## 現有優點（請保持不退化）

- `scriba/` 是 core library；可執行應用（API/Web/CLI）集中在 `apps/`
- `runs/<run_name>/` 是唯一實驗單位（可追溯、可重現）
- `models/<arch>/latest` 是預設模型指標（symlink），推論/比較預設都以此為準
- `apps/api` 與 `apps/web` 採 API-first（UI 只做呈現與互動）

## P0（先把研究流程跑順）

- [ ] 加入統一推論入口（`python -m apps.cli.infer`），支援 TrOCR/Donut、單張圖/資料夾、輸出 JSONL
- [ ] 改進頁面整體視覺設計與使用者體驗，優化介面的美觀度與易用性
- [ ] 加入 Web UI（FastAPI + Next.js，3 個頁：Inference / Comparison / Training Metrics）
  - [ ] Tab 1 Inference：單一模型推論，顯示更多資訊（source、latency、raw/json、可調 decoding 參數）
  - [ ] Tab 2 Comparison：預設 2 個模型，可按 + 增加到最多 4 個；同一 input 橫向比對輸出與 latency
  - [ ] Tab 3 Training Metrics：讀取 runs/<run_name>/ 的訓練紀錄（trainer_state.json 等），畫 line chart/table
  - [ ] Web 端支援 pretrained 一鍵下載到 `models/<arch>/<name>/{model,processor}`，並可設成 `models/<arch>/latest`
- [ ] 改進 training metrics 的同步機制與訓練流程的日誌記錄功能，並統一參數/指標/日誌條目的命名規範
- [ ] 補齊 models 慣例的文件與驗證（`models/<arch>/latest`）
- [ ] 將資料集格式寫成明確規範（CSV 欄位、相對路徑規則、split 命名）並在 `data/` 放範例
- [ ] 增加「小型 smoke dataset」與 smoke 測試（確保 train/eval 端到端不會壞）
- [ ] `runs/<run_name>/` 產物結構固定化（metrics.json、hparams.json、samples/）

## P1（研究擴展）

- [ ] 支援多資料集（不只 landlord），透過 `--dataset <name>` 切換
- [ ] 增加可插拔的 preprocessing（resize、normalize、augmentation、text normalization）
- [ ] 增加解碼策略（beam search、top-p、constraints）與可比較報表
- [ ] 加入實驗紀錄（wandb 可選 / 純本地 JSON 也可）

## P2（工程化與可維護）

- [ ] 補齊型別檢查/格式化工具（你想用 ruff/black/mypy 哪套）
- [ ] 加入基礎單元測試（metrics、dataset schema inference）
- [ ] 整理 requirements（CPU/GPU 分離、可選 extras）
- [ ] 封裝成可安裝套件（`pip install -e .`，是否要導入 `pyproject.toml`）
- [ ] Web UI 部署：支援單台 GPU server 多人使用（並發、資源/快取策略、日誌）
- [ ] API 認證：支援簡單 token（先）與 SSO/OAuth（後）

## 實驗想法（可直接做）

- [ ] Baseline compare：同一張圖比較 `trocr` vs `donut` 的文字正確率與 latency（先不用訓練）
- [ ] Decode sweep（TrOCR）：`num_beams`（1/3/5）× `max_new_tokens`（64/128/256），看 CER/WER vs latency tradeoff
- [ ] Text normalization：訓練/評測前做 lowercase、去重空白、移除標點（可選），觀察 CER/WER 變化
- [ ] Crop/resize 策略：固定長邊 resize vs padding，觀察 CER/WER 與速度
- [ ] Augment ablation：brightness/contrast/blur/rotate（小幅），看對泛化的幫助
- [ ] 低資料量曲線：抽樣 1%/5%/10%/100% 的 train set 訓練，畫學習曲線（CER/WER）
- [ ] Error bucket：把錯誤按字元類型（數字/英文/符號/空白）分 bucket，定位瓶頸
- [ ] 性能觀測：記錄每次 inference 的 latency（cpu/gpu）、image size、token length，做分佈圖

## 值得測試的模型（先用 pretrained 當 compare 範例）

### TrOCR（arch=trocr）

- [ ] `microsoft/trocr-base-handwritten`（目前預設）
- [ ] `microsoft/trocr-small-handwritten`（更快的 baseline）
- [ ] `microsoft/trocr-large-handwritten`（準確率可能更高）

### Donut（arch=donut）

- [ ] `naver-clova-ix/donut-base`（目前預設）
- [ ] `naver-clova-ix/donut-base-finetuned-cord-v2`（偏表單/收據結構化，適合當 structured JSON 對照）

### 之後可加（需要先接資料/格式）

- [ ] TrOCR + LoRA（peft），比較 full fine-tune vs LoRA 的收斂速度與效果
- [ ] LayoutLMv3 / LiLT（若你想做欄位/結構理解，會需要 bbox/版面資訊）

## 開放問題（需要你決定）

- [ ] 你希望「推論」的輸出格式是：純文字 / JSON（含 bbox、confidence）/ Donut style 的 structured JSON？
- [ ] `data/processed/` 要不要正式導入（例如把圖片 resize 後 cache 起來）？
- [ ] 以後模型會更多：想用 `scriba/models/<arch>/...` 固定模式，還是允許外掛註冊（registry plugin）？
