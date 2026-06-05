import type {
  CompareResponse,
  InferResponse,
  MetricSeriesResponse,
  ModelSpec,
  ModelsResponse,
  RunSummary,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function fetchModels(arch: string): Promise<ModelsResponse> {
  const url = new URL("/api/models", API_BASE);
  url.searchParams.set("arch", arch);
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as ModelsResponse;
}

export async function postInfer(args: {
  spec: ModelSpec;
  image: File;
  max_new_tokens: number;
  num_beams: number;
}): Promise<InferResponse> {
  const fd = new FormData();
  fd.set("arch", args.spec.arch);
  fd.set("source", args.spec.source);
  if (args.spec.stored_name) fd.set("stored_name", args.spec.stored_name);
  if (args.spec.pretrained_id) fd.set("pretrained_id", args.spec.pretrained_id);
  if (args.spec.custom_checkpoint_dir) fd.set("custom_checkpoint_dir", args.spec.custom_checkpoint_dir);
  fd.set("max_new_tokens", String(args.max_new_tokens));
  fd.set("num_beams", String(args.num_beams));
  fd.set("image", args.image);

  const res = await fetch(`${API_BASE}/api/infer`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as InferResponse;
}

export async function postCompare(args: {
  specs: ModelSpec[];
  image: File;
  max_new_tokens: number;
  num_beams: number;
}): Promise<CompareResponse> {
  const fd = new FormData();
  fd.set("specs_json", JSON.stringify(args.specs));
  fd.set("max_new_tokens", String(args.max_new_tokens));
  fd.set("num_beams", String(args.num_beams));
  fd.set("image", args.image);
  const res = await fetch(`${API_BASE}/api/compare`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as CompareResponse;
}

export async function fetchRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/api/runs`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as RunSummary[];
}

export async function fetchRunMetrics(run_id: string): Promise<MetricSeriesResponse> {
  const res = await fetch(`${API_BASE}/api/runs/${encodeURIComponent(run_id)}/metrics`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as MetricSeriesResponse;
}

