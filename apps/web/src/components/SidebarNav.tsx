"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { BarChart3, LayoutDashboard, ScanText } from "lucide-react";

import { cn } from "@/lib/cn";

const items = [
  { href: "/inference", label: "Inference", icon: ScanText },
  { href: "/compare", label: "Comparison", icon: LayoutDashboard },
  { href: "/runs", label: "Training Metrics", icon: BarChart3 },
];

export function SidebarNav() {
  const pathname = usePathname() || "/";

  return (
    <nav className="flex flex-col gap-1">
      {items.map((it) => {
        const active = pathname === it.href;
        const Icon = it.icon;
        return (
          <Link
            key={it.href}
            href={it.href}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition",
              active ? "bg-zinc-900 text-white" : "text-zinc-700 hover:bg-zinc-100",
            )}
          >
            <Icon size={16} />
            {it.label}
          </Link>
        );
      })}
    </nav>
  );
}

