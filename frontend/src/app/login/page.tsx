"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Zap } from "lucide-react";
import { api, tokens } from "@/lib/api";
import type { TokenPair } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, Field, Input } from "@/components/ui/primitives";
import { ErrorNotice } from "@/components/shared";
import { StormBolt } from "@/components/storm";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    try {
      const pair =
        mode === "login"
          ? await api.post<TokenPair>("/auth/login", { email: f.get("email"), password: f.get("password") })
          : await api.post<TokenPair>("/auth/register", {
              organization_name: f.get("organization_name"),
              full_name: f.get("full_name") || null,
              email: f.get("email"),
              password: f.get("password"),
            });
      tokens.set(pair);
      router.replace("/");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden px-4 py-10">
      <StormBolt className="absolute -top-6 right-[8%] h-[70vh] max-h-[560px] w-auto opacity-60 max-sm:right-[-12%]" />
      <StormBolt className="absolute bottom-[-8%] left-[4%] h-[38vh] max-h-[300px] w-auto -scale-x-100 opacity-40 max-sm:hidden" />
      <div className="relative w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="mb-4 inline-flex size-11 items-center justify-center rounded-lg bg-primary shadow-[0_0_28px_var(--glow)]">
            <Zap aria-hidden className="size-5 fill-primary-foreground text-primary-foreground" />
          </div>
          <p className="font-[family-name:var(--font-bricolage)] text-2xl font-semibold tracking-tight">StorePulse AI</p>
          <p className="mt-1 text-sm text-muted-foreground">Forecast demand and plan every store day.</p>
        </div>
        <Card className="bg-card/85 backdrop-blur">
          <CardContent className="p-6 sm:p-7">
            <h1 className="mb-1 text-xl font-semibold">{mode === "login" ? "Sign in" : "Create your workspace"}</h1>
            <p className="mb-6 text-sm text-muted-foreground">
              {mode === "login" ? "Welcome back." : "You'll be the owner of a new organization."}
            </p>
            <form onSubmit={onSubmit} className="flex flex-col gap-4">
              {mode === "register" ? (
                <>
                  <Field label="Organization">
                    <Input name="organization_name" required minLength={2} />
                  </Field>
                  <Field label="Your name">
                    <Input name="full_name" />
                  </Field>
                </>
              ) : null}
              <Field label="Email">
                <Input name="email" type="email" autoComplete="email" required />
              </Field>
              <Field label="Password" hint={mode === "register" ? "At least 10 characters." : undefined}>
                <Input
                  name="password"
                  type="password"
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  minLength={mode === "register" ? 10 : undefined}
                  required
                />
              </Field>
              <ErrorNotice error={error} />
              <Button type="submit" disabled={busy}>
                {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create workspace"}
              </Button>
            </form>
            <button
              type="button"
              className="mt-5 w-full rounded-sm text-center text-sm text-muted-foreground transition-colors duration-150 hover:text-primary"
              onClick={() => setMode(mode === "login" ? "register" : "login")}
            >
              {mode === "login" ? "New here? Create a workspace" : "Already have an account? Sign in"}
            </button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
