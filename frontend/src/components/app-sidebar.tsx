"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Activity, BarChart3, Database, FileText, FlaskConical, LogOut, Rocket, Store, Wallet, Zap } from "lucide-react";
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
    <aside className="charged-edge sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-sidebar text-sidebar-foreground md:flex">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="flex size-8 items-center justify-center rounded-md bg-primary shadow-[0_0_18px_var(--glow)]">
          <Zap aria-hidden className="size-4 fill-primary-foreground text-primary-foreground" />
        </div>
        <span className="font-[family-name:var(--font-bricolage)] text-base font-semibold tracking-tight">StorePulse AI</span>
      </div>
      <nav aria-label="Main" className="flex flex-1 flex-col gap-1 px-3">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "relative flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors duration-150",
                "focus-visible:outline-offset-0",
                active
                  ? "bg-sidebar-active font-medium text-white before:absolute before:inset-y-1.5 before:left-0 before:w-0.5 before:rounded-full before:bg-linear-to-b before:from-primary before:to-accent-violet before:shadow-[0_0_10px_var(--glow)]"
                  : "text-sidebar-muted hover:bg-sidebar-active/60 hover:text-white",
              )}
            >
              <Icon aria-hidden className={cn("size-4", active && "text-primary")} />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-white/10 px-5 py-4 text-xs">
        <p className="truncate font-medium">{me?.organization.name ?? "…"}</p>
        <p className="truncate text-sidebar-muted">{me?.user.email}</p>
        <button
          className="mt-3 flex items-center gap-2 rounded-sm text-sidebar-muted transition-colors duration-150 hover:text-white"
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
    <nav aria-label="Main" className="sticky top-0 z-20 flex gap-1 overflow-x-auto border-b bg-sidebar/90 px-4 py-2 backdrop-blur md:hidden">
      {NAV.map(({ href, label }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs whitespace-nowrap transition-colors duration-150",
              active
                ? "bg-primary/15 font-medium text-primary shadow-[inset_0_0_0_1px_var(--glow)]"
                : "text-muted-foreground hover:bg-muted hover:text-foreground",
            )}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
