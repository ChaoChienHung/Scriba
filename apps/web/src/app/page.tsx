import Link from "next/link";

export default function Home() {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-6 py-10">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">Scriba Web</h1>
        <p className="text-sm text-zinc-600">
          OCR research sandbox：單模型推論、模型對照、訓練 metrics dashboard
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Link
          href="/inference"
          className="rounded-xl border border-zinc-200 bg-white p-5 transition hover:border-zinc-300 hover:shadow-sm"
        >
          <div className="text-sm font-medium text-zinc-700">Inference</div>
          <div className="mt-1 text-lg font-semibold">單模型推論</div>
          <div className="mt-2 text-sm text-zinc-600">
            上傳圖片、選模型來源、調 decoding 參數，查看輸出與 latency
          </div>
        </Link>

        <Link
          href="/compare"
          className="rounded-xl border border-zinc-200 bg-white p-5 transition hover:border-zinc-300 hover:shadow-sm"
        >
          <div className="text-sm font-medium text-zinc-700">Comparison</div>
          <div className="mt-1 text-lg font-semibold">2~4 模型對照</div>
          <div className="mt-2 text-sm text-zinc-600">
            同一張圖橫向比對不同模型輸出與 latency
          </div>
        </Link>

        <Link
          href="/runs"
          className="rounded-xl border border-zinc-200 bg-white p-5 transition hover:border-zinc-300 hover:shadow-sm"
        >
          <div className="text-sm font-medium text-zinc-700">Training Metrics</div>
          <div className="mt-1 text-lg font-semibold">訓練指標</div>
          <div className="mt-2 text-sm text-zinc-600">
            讀取 runs/*/trainer_state.json，畫出 loss/指標走勢
          </div>
        </Link>
      </div>
    </div>
  );
}
