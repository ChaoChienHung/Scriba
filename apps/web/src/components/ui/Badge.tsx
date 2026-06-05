"use client";

import type { HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type Variant = "default" | "success" | "warning";

type Props = HTMLAttributes<HTMLSpanElement> & {
  variant?: Variant;
};

export function Badge({ className, variant = "default", ...props }: Props) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium",
        variant === "default" ? "bg-zinc-100 text-zinc-700" : null,
        variant === "success" ? "bg-emerald-100 text-emerald-700" : null,
        variant === "warning" ? "bg-amber-100 text-amber-800" : null,
        className,
      )}
      {...props}
    />
  );
}

