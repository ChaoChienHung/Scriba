"use client";

import { useMemo, useState } from "react";

import Image from "next/image";

import { ModelSelector } from "@/components/ModelSelector";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { postInfer } from "@/lib/api";
import type { InferResponse, ModelSpec } from "@/lib/types";

export default function InferencePage() {
  const [spec, setSpec] = useState<ModelSpec>({ arch: "trocr", source: "latest" });
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [numBeams, setNumBeams] = useState(1);
  const [image, setImage] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<InferResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const previewUrl = useMemo(() => (image ? URL.createObjectURL(image) : null), [image]);

  async function onRun() {
    if (!image) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await postInfer({
        spec,
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
          <h2 className="text-xl font-semibold tracking-tight">Inference</h2>
          <div className="flex items-center gap-2">
            {result ? <Badge variant="default">{result.source_resolved.kind}</Badge> : null}
            {result ? <Badge variant="default">{result.latency_ms.toFixed(1)} ms</Badge> : null}
          </div>
        </div>
        <p className="text-sm text-zinc-600">單一模型推論：模型來源 / decoding 參數 / 輸出與 latency</p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-4">
          <ModelSelector value={spec} onChange={setSpec} />

          <Card>
            <CardHeader>
              <div>
                <CardTitle>Input</CardTitle>
                <CardDescription>上傳圖片並調整 generation 參數</CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <label className="flex flex-col gap-1 text-sm">
                <span className="text-zinc-700">Image</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setImage(e.target.files?.[0] || null)}
                />
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
            </div>

              <div className="mt-4 flex items-center gap-2">
                <Button disabled={!image || busy} onClick={onRun}>
                  {busy ? "Running..." : "Run inference"}
                </Button>
                <Button
                  variant="secondary"
                  disabled={!result || busy}
                  onClick={async () => {
                    if (!result) return;
                    await navigator.clipboard.writeText(JSON.stringify(result.output, null, 2));
                    setCopied(true);
                    setTimeout(() => setCopied(false), 900);
                  }}
                >
                  {copied ? "Copied" : "Copy JSON"}
                </Button>
              </div>

              {err ? <div className="mt-3 text-sm text-red-600">{err}</div> : null}
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Preview</CardTitle>
                <CardDescription>確認輸入圖片與解析度</CardDescription>
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
                  className="max-h-[360px] w-auto rounded-md object-contain"
                  unoptimized
                />
              ) : (
                <div className="text-sm text-zinc-500">Upload an image to preview</div>
              )}
            </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div>
                <CardTitle>Result</CardTitle>
                <CardDescription>輸出（trocr:text / donut:raw+json）</CardDescription>
              </div>
              {result ? <Badge variant="default">{result.source_resolved.value}</Badge> : null}
            </CardHeader>

            {result ? (
              <CardContent>
                <pre className="max-h-[420px] overflow-auto rounded-lg bg-zinc-950 p-3 text-xs text-zinc-50">
                  {JSON.stringify(result.output, null, 2)}
                </pre>
              </CardContent>
            ) : (
              <CardContent>
                <div className="text-sm text-zinc-500">No result yet</div>
              </CardContent>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
