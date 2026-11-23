"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Activity, BarChart3, Database, FileText, FlaskConical, LogOut, Rocket, Store, Wallet } from "lucide-react";
import { tokens } from "@/lib/api";
import { useMe } from "@/hooks/queries";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Command Center", icon: Activity },
  { href: "/data", label: "Data & EDA", icon: Database },
  { href: "/forecasts", label: "Forecasts", icon: BarChart3 },
  { href: "/model-lab", label: "Model Lab", icon: FlaskConical },
  { href: "/autopilot", label: "Store Autopilot", icon: Rocket },
  { href: "/pnl-workforce", label: "P&L & Workforce", icon: Wallet },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/stores", label: "Stores", icon: Store },
];

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { data: me } = useMe();

  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-sidebar text-sidebar-foreground md:flex">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="flex size-7 items-center justify-center rounded-md bg-primary">
          <BarChart3 className="size-4 text-primary-foreground" />
        </div>
        <span className="text-sm font-semibold tracking-tight">StorePulse AI</span>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 px-3">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                active ? "bg-sidebar-active text-white" : "text-sidebar-muted hover:bg-sidebar-active/60 hover:text-white",
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-white/10 px-5 py-4 text-xs">
        <p className="truncate font-medium">{me?.organization.name ?? "…"}</p>
        <p className="truncate text-sidebar-muted">{me?.user.email}</p>
        <button
          className="mt-3 flex items-center gap-2 text-sidebar-muted hover:text-white"
          onClick={() => {
            tokens.clear();
            router.replace("/login");
          }}
        >
          <LogOut className="size-3.5" /> Sign out
        </button>
      </div>
    </aside>
  );
}

export function MobileNav() {
  const pathname = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto border-b bg-card px-4 py-2 md:hidden">
      {NAV.map(({ href, label }) => (
        <Link
          key={href}
          href={href}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs whitespace-nowrap",
            (href === "/" ? pathname === "/" : pathname.startsWith(href)) ? "bg-muted font-medium" : "text-muted-foreground",
          )}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
