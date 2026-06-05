"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchModels } from "@/lib/api";
import type { ModelSpec, ModelsResponse } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";

type Props = {
  value: ModelSpec;
  onChange: (next: ModelSpec) => void;
  title?: string;
};

export function ModelSelector({ value, onChange, title }: Props) {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    queueMicrotask(() => {
      if (!alive) return;
      setModels(null);
      setErr(null);
    });
    fetchModels(value.arch)
      .then((m) => {
        if (alive) setModels(m);
      })
      .catch((e) => {
        if (alive) setErr(String(e?.message || e));
      });
    return () => {
      alive = false;
    };
  }, [value.arch]);

  const storedNames = useMemo(() => models?.stored?.map((s) => s.name) || [], [models]);

  return (
    <Card>
      <CardHeader>
        <div className="min-w-0">
          <CardTitle>{title || "Model"}</CardTitle>
          <CardDescription>arch + source + checkpoint/pretrained 選擇</CardDescription>
        </div>
        {models ? (
          <div className="flex shrink-0 items-center gap-2">
            <Badge variant={models.latest_exists ? "success" : "default"}>latest</Badge>
            <Badge variant="default">stored {models.stored.length}</Badge>
          </div>
        ) : err ? (
          <Badge variant="warning">models error</Badge>
        ) : (
          <Badge variant="default">loading</Badge>
        )}
      </CardHeader>

      <CardContent className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-700">Arch</span>
            <Select value={value.arch} onChange={(e) => onChange({ ...value, arch: e.target.value as ModelSpec["arch"] })}>
              <option value="trocr">trocr</option>
              <option value="donut">donut</option>
            </Select>
          </label>

          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-700">Source</span>
            <Select
              value={value.source}
              onChange={(e) => onChange({ ...value, source: e.target.value as ModelSpec["source"] })}
            >
              <option value="latest">latest</option>
              <option value="stored">stored</option>
              <option value="pretrained">pretrained</option>
              <option value="custom">custom</option>
            </Select>
          </label>
        </div>

        {value.source === "stored" ? (
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-700">Stored Model</span>
            <Select value={value.stored_name || ""} onChange={(e) => onChange({ ...value, stored_name: e.target.value })}>
              <option value="" disabled>
                {storedNames.length ? "Select..." : "No stored models"}
              </option>
              {storedNames.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </Select>
          </label>
        ) : null}

        {value.source === "pretrained" ? (
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-700">Pretrained ID</span>
            <Input
              value={value.pretrained_id || ""}
              placeholder={value.arch === "donut" ? "naver-clova-ix/donut-base" : "microsoft/trocr-base-handwritten"}
              onChange={(e) => onChange({ ...value, pretrained_id: e.target.value })}
            />
          </label>
        ) : null}

        {value.source === "custom" ? (
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-zinc-700">Custom Checkpoint Dir</span>
            <Input
              value={value.custom_checkpoint_dir || ""}
              placeholder="runs/xxx or models/trocr/xxx"
              onChange={(e) => onChange({ ...value, custom_checkpoint_dir: e.target.value })}
            />
          </label>
        ) : null}

        {err ? <div className="text-xs text-red-600">{err}</div> : null}
      </CardContent>
    </Card>
  );
}
