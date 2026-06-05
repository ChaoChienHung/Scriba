"use client";

import type { SelectHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type Props = SelectHTMLAttributes<HTMLSelectElement>;

export function Select({ className, ...props }: Props) {
  return (
    <select
      className={cn(
        "h-10 w-full rounded-lg border border-zinc-200 bg-white px-3 text-sm text-zinc-900 outline-none",
        "focus:border-zinc-400 focus:ring-2 focus:ring-zinc-900/10",
        className,
      )}
      {...props}
    />
  );
}

