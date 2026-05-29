# TODO

## P0（先把研究流程跑順）

- [ ] 加入統一推論入口（`python -m scriba.infer`），支援 TrOCR/Donut、單張圖/資料夾、輸出 JSONL
- [ ] 加入 Streamlit Web UI（比較頁 + runs/metrics/perf dashboard）
- [ ] 定義並落地 models 慣例（`models/<arch>/latest`）
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

## 開放問題（需要你決定）

- [ ] 你希望「推論」的輸出格式是：純文字 / JSON（含 bbox、confidence）/ Donut style 的 structured JSON？
- [ ] `data/processed/` 要不要正式導入（例如把圖片 resize 後 cache 起來）？
- [ ] 以後模型會更多：想用 `scriba/models/<arch>/...` 固定模式，還是允許外掛註冊（registry plugin）？
