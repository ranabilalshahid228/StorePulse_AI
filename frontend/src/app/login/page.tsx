"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { BarChart3 } from "lucide-react";
import { api, tokens } from "@/lib/api";
import type { TokenPair } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, Field, Input } from "@/components/ui/primitives";
import { ErrorNotice } from "@/components/shared";

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
    <div className="flex min-h-screen items-center justify-center bg-sidebar px-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex items-center justify-center gap-2 text-white">
          <div className="flex size-8 items-center justify-center rounded-md bg-primary">
            <BarChart3 className="size-4" />
          </div>
          <span className="text-lg font-semibold">StorePulse AI</span>
        </div>
        <Card>
          <CardContent className="p-6">
            <h1 className="mb-1 text-base font-semibold">{mode === "login" ? "Sign in" : "Create your workspace"}</h1>
            <p className="mb-5 text-xs text-muted-foreground">
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
              className="mt-4 w-full text-center text-xs text-muted-foreground hover:text-foreground"
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
