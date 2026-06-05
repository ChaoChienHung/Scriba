"use client";

import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
};

export function Button({ className, variant = "primary", size = "md", ...props }: Props) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg font-medium transition outline-none disabled:pointer-events-none disabled:opacity-50",
        "focus-visible:ring-2 focus-visible:ring-zinc-900/10 focus-visible:ring-offset-2 focus-visible:ring-offset-white",
        size === "sm" ? "h-9 px-3 text-sm" : "h-10 px-4 text-sm",
        variant === "primary" ? "bg-zinc-900 text-white hover:bg-zinc-800" : null,
        variant === "secondary" ? "border border-zinc-200 bg-white text-zinc-900 hover:bg-zinc-50" : null,
        variant === "ghost" ? "text-zinc-700 hover:bg-zinc-100" : null,
        variant === "danger" ? "bg-red-600 text-white hover:bg-red-500" : null,
        className,
      )}
      {...props}
    />
  );
}

