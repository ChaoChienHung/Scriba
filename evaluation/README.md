# Evaluation

這個資料夾用來收納「可版本控制、可長期累積」的評估與對照資產（不是每次訓練 run 的一次性輸出）。

- `evaluation/eval_sets/`：離線評估用的固定測資（例如 small smoke set、固定對照圖片清單、或格式化輸出規範）
- `evaluation/results/model_answers/`：針對固定測資的模型輸出（可用於長期對照與回歸測試）
- `evaluation/results/judge/`：若未來加入人工或 LLM-as-a-judge 的標註/排行榜，放在這裡
- `evaluation/logs/`：跑 benchmark / 評估命令時的 stdout logs（便於排錯）

一次性評估輸出仍以 `runs/<run_name>/eval/<split>/` 為準（見 `AGENTS.md`）。

