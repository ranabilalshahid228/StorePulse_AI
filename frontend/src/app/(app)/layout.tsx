"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { AppSidebar, MobileNav } from "@/components/app-sidebar";
import { tokens } from "@/lib/api";

export default function AppLayout({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!tokens.access) router.replace("/login");
    else setReady(true);
  }, [router]);

  if (!ready) return null;
  return (
    <div className="flex min-h-screen">
      <AppSidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <MobileNav />
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 md:px-8">{children}</main>
      </div>
    </div>
  );
}
