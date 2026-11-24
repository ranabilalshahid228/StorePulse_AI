import { cn } from "@/lib/utils";

/** Decorative branching lightning bolt (hidden from assistive tech). It glows slowly via `.storm-bolt`
 *  and stays still when the visitor asks for reduced motion. */
export function StormBolt({ className }: { className?: string }) {
  return (
    <svg
      aria-hidden
      focusable="false"
      viewBox="0 0 120 220"
      fill="none"
      className={cn("storm-bolt pointer-events-none", className)}
    >
      <defs>
        <linearGradient id="storm-bolt-stroke" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="var(--primary)" />
          <stop offset="1" stopColor="var(--accent-violet)" />
        </linearGradient>
      </defs>
      <g stroke="url(#storm-bolt-stroke)" strokeLinecap="round" strokeLinejoin="round">
        <path d="M68 2 L52 58 L70 64 L44 128 L62 134 L30 218" strokeWidth="2.4" />
        <path d="M52 58 L34 82 L38 96" strokeWidth="1.1" opacity="0.7" />
        <path d="M70 64 L92 92 L88 112 L102 132" strokeWidth="1.1" opacity="0.6" />
        <path d="M44 128 L24 150" strokeWidth="0.9" opacity="0.55" />
        <path d="M62 134 L80 162 L76 178" strokeWidth="0.9" opacity="0.5" />
      </g>
    </svg>
  );
}
