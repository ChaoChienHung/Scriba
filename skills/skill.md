# Karen Skill

這份文件把這個 repo 當成一個「可重複使用的本地 LLM 實驗 skill」：用同一套資料管線與統一入口，快速切換 pretrained fine-tune（HuggingFace + LoRA）與 scratch 小模型（GPT2/Llama）。

## 你可以用它做什麼

- 用同一個入口訓練不同架構：`python -m llm.train`
- 對同一個 run 進行評估：`python -m llm.eval --run_dir runs/<run_name>`
- 產出可追溯的 artifacts：args/env/logs/metrics/weights/tokenizer、eval 報表（`runs/<run_name>/eval/*`）
- LoRA adapter 合併回 base model：`python -m llm.tools.merge_lora`

## Quickstart

```bash
pip install -r requirements.txt
python scripts/download/download_dolly.py --dataset_name databricks/databricks-dolly-15k
```

### Pretrained Fine-tune（預設）

```bash
python -m llm.train --arch Llama-2-7b --engine trainer --dataset_path data/raw/dolly
```

### Pretrained GPT2

```bash
python -m llm.train --arch gpt2 --engine trainer --dataset_path data/raw/dolly
```

### Scratch 小模型（dev）

```bash
python -m llm.train --arch gpt2-dev --engine trainer --dataset_path data/raw/dolly --scratch_d_model 256 --scratch_n_layer 6 --scratch_n_head 8
python -m llm.train --arch llama-dev --engine loop --dataset_path data/raw/dolly --scratch_d_model 256 --scratch_n_layer 6 --scratch_n_head 8
```

## Evaluation（統一評估入口）

```bash
python -m llm.eval --run_dir runs/<run_name>
```

預期輸出（依設定略有差異）：

- `runs/<run_name>/eval/metrics.json`
- `runs/<run_name>/eval/instruction_eval.jsonl`
- `runs/<run_name>/eval/exact_match_eval.jsonl`
- `runs/<run_name>/eval/smoke_generations.jsonl`

Benchmark 設計與實驗記錄：`DevNotes.md`

## Outputs / Checkpoints（常用路徑）

- Common run artifacts（兩個 engine 一致）
  - Run config：`runs/<run_name>/args.json`
  - Environment snapshot：`runs/<run_name>/env.json`
  - Logs：`runs/<run_name>/train.rank<RANK>.log`
  - Metrics stream：`runs/<run_name>/metrics.jsonl`
- Trainer engine（`--engine trainer`）
  - Checkpoints：`runs/<run_name>/checkpoint-*`
  - Final weights
    - LoRA：`runs/<run_name>/adapter/`
    - Non-LoRA：`runs/<run_name>/model/`
  - Tokenizer：`runs/<run_name>/`
- Custom loop（`--engine loop`）
  - Final weights：`runs/<run_name>/model/`
  - Tokenizer：`runs/<run_name>/`

## Wandb（可選，預設關閉）

用 flag 啟用：

```bash
python -m llm.train ... --wandb --wandb_project llm-lab
```

或用環境變數啟用：

```bash
WANDB_ENABLED=1 python -m llm.train ...
```

## LoRA Merge（推論/部署前常用）

```bash
python -m llm.tools.merge_lora --adapter_dir runs/<run_name>/adapter --output_dir model/merged
```

## 建議工作流（最小可行）

1. 下載或準備資料集 → `--dataset_path`
2. 訓練 → `python -m llm.train ... --run_name <name>`
3. 評估 → `python -m llm.eval --run_dir runs/<name>`
4. 需要推論或部署 → merge LoRA 或直接使用輸出權重

## TODO 導航

接下來的工程化與研究方向集中在 `TODO.md`（例如：tokenizer、custom loop（AMP/DDP）、evaluation pipeline、單測、lint/CI、License）。 

