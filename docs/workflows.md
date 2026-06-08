# Workflows

本文件把常見工作流整理成可直接照做的步驟，並標示「必備產物」與「驗收點」。不可退化約束以 [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md) 為準。

## 1) Train（訓練）

- 入口：`python3 -m scriba.train`
- 必備產物（落地到 `runs/<run_name>/`）
  - `config.json`
  - `model/`
  - `processor/`
- 常用範例：

```bash
python3 -m scriba.train --arch trocr --publish-latest
python3 -m scriba.train --arch donut --max-target-length 256 --publish-latest
```

## 2) Evaluate（評估）

- 入口：`python3 -m scriba.eval`
- 若指定 `--run-dir runs/<run_name>`，必須回寫：
  - `runs/<run_name>/eval/<split>/metrics.json`
  - （可選）`runs/<run_name>/eval/<split>/predictions.jsonl`
- 常用範例：

```bash
python3 -m scriba.eval --arch trocr --split test
python3 -m scriba.eval --arch donut --split test
python3 -m scriba.eval --arch trocr --model-dir runs/<run_name>/model --split test
```

## 3) Download Pretrained（下載預訓練權重到本地）

- 入口：`python3 -m apps.cli.download`
- 產物：`models/<arch>/<name>/{model,processor}`
- 可選：更新 `models/<arch>/latest` 指向（對推論/比較作為預設）

```bash
python3 -m apps.cli.download --arch trocr --pretrained microsoft/trocr-base-handwritten --set-latest
python3 -m apps.cli.download --arch donut --pretrained naver-clova-ix/donut-base --set-latest
```

## 4) Inference（推論）

### CLI

```bash
python3 -m apps.cli.infer --arch trocr --image path/to/image.jpg
python3 -m apps.cli.infer --arch donut --image path/to/image.jpg
```

### Web UI

```bash
cd apps/web
npm run dev -- --port 3000
```

### API

```bash
python3 -m uvicorn apps.api.app.main:app --reload --port 8000
```

## 5) 模型生命週期（latest 指標）

- `models/<arch>/latest` 是「預設模型入口」
- 建議操作：
  - 開發/比較：用 `apps.cli.download --set-latest` 或訓練 `--publish-latest`
  - 需要回滾：把 symlink 指回上一個已知可用的 checkpoint 目錄
- 驗收點：
  - CLI/Web/API 在未額外指定 `--model-dir` 時，能自動使用 `latest` 完成推論
