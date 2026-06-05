"use client";

import Image from "next/image";
import { useMemo, useState } from "react";

import { ModelSelector } from "@/components/ModelSelector";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { postCompare } from "@/lib/api";
import type { CompareResponse, ModelSpec } from "@/lib/types";

const labels = ["A", "B", "C", "D"] as const;

function defaultSpec(i: number): ModelSpec {
  if (i === 0) return { arch: "trocr", source: "latest" };
  return { arch: "donut", source: "latest" };
}

export default function ComparePage() {
  const [n, setN] = useState(2);
  const [specs, setSpecs] = useState<ModelSpec[]>([defaultSpec(0), defaultSpec(1)]);
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [numBeams, setNumBeams] = useState(1);
  const [image, setImage] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);

  const previewUrl = useMemo(() => (image ? URL.createObjectURL(image) : null), [image]);

  function setSpecAt(idx: number, next: ModelSpec) {
    setSpecs((prev) => prev.map((s, i) => (i === idx ? next : s)));
  }

  function addModel() {
    setN((prev) => {
      const nextN = Math.min(4, prev + 1);
      setSpecs((s) => (s.length >= nextN ? s : [...s, defaultSpec(s.length)]));
      return nextN;
    });
  }

  function removeModel() {
    setN((prev) => {
      const nextN = Math.max(2, prev - 1);
      setSpecs((s) => s.slice(0, nextN));
      return nextN;
    });
  }

  async function onRun() {
    if (!image) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await postCompare({
        specs: specs.slice(0, n),
        image,
        max_new_tokens: maxNewTokens,
        num_beams: numBeams,
      });
      setResult(r);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-8">
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold tracking-tight">Comparison</h2>
          <div className="flex items-center gap-2">
            <Badge variant="default">{n} models</Badge>
            {result ? <Badge variant="default">{result.request_id.slice(0, 8)}</Badge> : null}
          </div>
        </div>
        <p className="text-sm text-zinc-600">同一 input 橫向比對 2~4 個模型輸出與 latency</p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Input</CardTitle>
                <CardDescription>上傳圖片並設定共同的 decoding 參數</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-3">
              <label className="flex flex-col gap-1 text-sm">
                <span className="text-zinc-700">Image</span>
                <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] || null)} />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="text-zinc-700">max_new_tokens</span>
                <Input
                  type="number"
                  min={1}
                  max={2048}
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                />
              </label>
              <label className="flex flex-col gap-1 text-sm">
                <span className="text-zinc-700">num_beams</span>
                <Input
                  type="number"
                  min={1}
                  max={16}
                  value={numBeams}
                  onChange={(e) => setNumBeams(Number(e.target.value))}
                />
              </label>

              <div className="flex items-center gap-2">
                <Button variant="secondary" size="sm" onClick={removeModel} disabled={n <= 2 || busy}>
                  − Remove
                </Button>
                <Button variant="secondary" size="sm" onClick={addModel} disabled={n >= 4 || busy}>
                  + Add
                </Button>
              </div>

              <Button disabled={!image || busy} onClick={onRun}>
                {busy ? "Running..." : "Run comparison"}
              </Button>

              {err ? <div className="text-sm text-red-600">{err}</div> : null}
            </div>
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardHeader>
              <div>
                <CardTitle>Preview</CardTitle>
                <CardDescription>確認輸入圖片</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center rounded-lg border border-dashed border-zinc-200 bg-zinc-50 p-3">
              {previewUrl ? (
                <Image
                  src={previewUrl}
                  alt="preview"
                  width={1200}
                  height={800}
                  className="max-h-[260px] w-auto rounded-md object-contain"
                  unoptimized
                />
              ) : (
                <div className="text-sm text-zinc-500">Upload an image to preview</div>
              )}
            </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {specs.slice(0, n).map((s, i) => (
              <ModelSelector key={i} title={`Model ${labels[i]}`} value={s} onChange={(next) => setSpecAt(i, next)} />
            ))}
          </div>

          <Card className="mt-4">
            <CardHeader>
              <div>
                <CardTitle>Results</CardTitle>
                <CardDescription>每個模型一張卡片：source / latency / output</CardDescription>
              </div>
            </CardHeader>

            {result ? (
              <CardContent>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {result.results.map((r, i) => (
                  <div key={i} className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="text-sm font-semibold text-zinc-900">{labels[i]}</div>
                        <Badge variant="default">{r.arch}</Badge>
                      </div>
                      <Badge variant="default">{r.latency_ms.toFixed(1)} ms</Badge>
                    </div>
                    <div className="mt-2 text-xs text-zinc-600">{r.source_resolved.value}</div>
                    <pre className="mt-3 max-h-[260px] overflow-auto rounded-lg bg-zinc-950 p-3 text-xs text-zinc-50">
                      {JSON.stringify(r.output, null, 2)}
                    </pre>
                  </div>
                ))}
                </div>
              </CardContent>
            ) : (
              <CardContent>
                <div className="text-sm text-zinc-500">No comparison yet</div>
              </CardContent>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
