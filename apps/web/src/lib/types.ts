export type Arch = "trocr" | "donut";
export type ModelSource = "latest" | "stored" | "pretrained" | "custom";

export type StoredModelItem = {
  name: string;
  path: string;
};

export type ModelsResponse = {
  arch: string;
  latest_exists: boolean;
  stored: StoredModelItem[];
};

export type InferResponse = {
  request_id: string;
  arch: string;
  source_resolved: {
    kind: "checkpoint_dir" | "pretrained";
    value: string;
  };
  latency_ms: number;
  output: Record<string, unknown>;
};

export type CompareResponse = {
  request_id: string;
  results: InferResponse[];
};

export type RunSummary = {
  run_id: string;
  path: string;
  has_trainer_state: boolean;
};

export type MetricPoint = {
  step?: number;
  epoch?: number;
  timestamp?: number;
  values: Record<string, unknown>;
};

export type MetricSeriesResponse = {
  run_id: string;
  keys: string[];
  series: MetricPoint[];
};

export type ModelSpec = {
  arch: Arch;
  source: ModelSource;
  stored_name?: string;
  pretrained_id?: string;
  custom_checkpoint_dir?: string;
};

