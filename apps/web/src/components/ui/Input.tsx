"use client";

import type { InputHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type Props = InputHTMLAttributes<HTMLInputElement>;

export function Input({ className, ...props }: Props) {
  return (
    <input
      className={cn(
        "h-10 w-full rounded-lg border border-zinc-200 bg-white px-3 text-sm text-zinc-900 outline-none placeholder:text-zinc-400",
        "focus:border-zinc-400 focus:ring-2 focus:ring-zinc-900/10",
        className,
      )}
      {...props}
    />
  );
}

