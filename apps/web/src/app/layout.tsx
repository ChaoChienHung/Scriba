import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

import { SidebarNav } from "@/components/SidebarNav";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Scriba",
  description: "OCR research sandbox",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-zinc-100 text-zinc-950">
        <div className="flex min-h-dvh">
          <aside className="hidden w-64 shrink-0 border-r border-zinc-200 bg-white px-4 py-6 lg:block">
            <Link href="/" className="px-3 text-lg font-semibold tracking-tight text-zinc-950">
              Scriba
            </Link>
            <div className="mt-6">
              <SidebarNav />
            </div>
          </aside>
          <div className="flex min-w-0 flex-1 flex-col">
            <header className="border-b border-zinc-200 bg-white lg:hidden">
              <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
                <Link href="/" className="text-lg font-semibold tracking-tight">
                  Scriba
                </Link>
                <div className="text-xs font-medium text-zinc-500">OCR sandbox</div>
              </div>
            </header>
            <main className="flex-1">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
