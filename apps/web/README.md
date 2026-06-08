# Scriba Web (Next.js)

Scriba 的 Web UI（Inference / Comparison / Training Metrics）。整體專案入口與規範請先看根目錄：

- Repo guardrails: [AGENTS.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/AGENTS.md)
- Docs map: [docs/doc-map.md](file:///Users/bytedance/Desktop/Ludwig/Scriba/docs/doc-map.md)

## Local Dev

```bash
npm install
npm run dev -- --port 3000
```

## Backend API（FastAPI）

在 repo 根目錄啟動：

```bash
python3 -m uvicorn apps.api.app.main:app --reload --port 8000
```
