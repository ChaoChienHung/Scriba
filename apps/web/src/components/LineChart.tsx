"use client";

import { useMemo } from "react";

type Props = {
  values: number[];
  width?: number;
  height?: number;
};

export function LineChart({ values, width = 760, height = 220 }: Props) {
  const { points, yMin, yMax } = useMemo(() => {
    if (!values.length) return { points: "", yMin: 0, yMax: 0 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const padX = 8;
    const padY = 10;
    const w = width - padX * 2;
    const h = height - padY * 2;
    const pts = values
      .map((v, i) => {
        const x = padX + (values.length === 1 ? 0 : (i / (values.length - 1)) * w);
        const y = padY + (1 - (v - min) / span) * h;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
    return { points: pts, yMin: min, yMax: max };
  }, [values, width, height]);

  return (
    <div className="w-full overflow-x-auto rounded-xl border border-zinc-200 bg-white p-3 shadow-sm">
      <div className="flex items-center justify-between px-1 pb-2 text-xs text-zinc-600">
        <span>
          min {Number.isFinite(yMin) ? yMin.toFixed(4) : "-"} · max{" "}
          {Number.isFinite(yMax) ? yMax.toFixed(4) : "-"}
        </span>
        <span>{values.length} points</span>
      </div>
      <svg width={width} height={height} className="block">
        <rect x={0} y={0} width={width} height={height} fill="transparent" />
        <polyline points={points} fill="none" stroke="#0f172a" strokeWidth={2} strokeLinejoin="round" />
      </svg>
    </div>
  );
}
