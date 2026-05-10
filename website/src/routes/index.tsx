import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Copy, Check, Mic, Volume2, Waves, Compass, Zap, Brain,
  Globe, Monitor, Terminal, Keyboard, Code2, Database,
  LayoutGrid, ArrowRight, ArrowDown,
  Scale, Briefcase, HeartPulse, Cpu,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import logo from "@/assets/sensus-logo.png";

export const Route = createFileRoute("/")({
  component: Index,
});

const TECHS = [
  "Rust", "Python", "Whisper", "LLaMA", "Ollama", "D-Bus",
  "systemd", "Wayland", "PipeWire", "GTK", "PyTorch", "ONNX",
];

function SoundWaves() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = 0, H = 0, dpr = 1;

    const resize = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = canvas.clientWidth;
      H = canvas.clientHeight;
      if (W === 0 || H === 0) return;
      canvas.width = Math.floor(W * dpr);
      canvas.height = Math.floor(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const start = performance.now();
    let raf = 0;

    // Pseudo-random per-bar offsets for organic motion
    const seed = (i: number) => {
      const s = Math.sin(i * 12.9898) * 43758.5453;
      return s - Math.floor(s);
    };

    const draw = () => {
      if (W === 0 || H === 0) {
        raf = requestAnimationFrame(draw);
        return;
      }
      const t = (performance.now() - start) / 1000;
      ctx.clearRect(0, 0, W, H);
      const fg = getComputedStyle(canvas).color || "#000";
      ctx.fillStyle = fg;

      const barW = 2;
      const gap = 6;
      const stride = barW + gap;
      const cy = H / 2;
      const maxH = H * 0.55;

      // Multiple sine layers combined for an organic audio-waveform shape
      for (let i = 0, x = 0; x < W; i++, x += stride) {
        const u = x / W;
        const edge = Math.pow(Math.sin(u * Math.PI), 0.6); // soft edges only

        // Layered traveling sines — like a real waveform
        const w1 = Math.sin(u * 9.0 + t * 1.1);
        const w2 = Math.sin(u * 21.0 - t * 0.7 + 1.3);
        const w3 = Math.sin(u * 43.0 + t * 1.9 + 2.1);
        const w4 = Math.sin(u * 4.0 - t * 0.4);
        const wave = (w1 * 0.55 + w2 * 0.3 + w3 * 0.18 + w4 * 0.4);

        // Per-bar tiny offset to break perfect periodicity
        const r = seed(i);
        const jitter = 0.9 + r * 0.2;

        const amp = Math.abs(wave) * jitter;
        const h = Math.max(1, maxH * edge * amp);

        const alpha = 0.04 + 0.14 * edge * (0.4 + 0.6 * Math.abs(wave));
        ctx.globalAlpha = Math.min(0.22, alpha);
        ctx.fillRect(x, cy - h / 2, barW, h);
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full text-foreground"
      aria-hidden
    />
  );
}

function InstallLine() {
  const cmd = "git clone https://github.com/HarisK03/sensus";
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(cmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.7, duration: 0.7 }}
      className="mt-10 flex w-full max-w-xl items-center justify-center gap-3 rounded-xl border border-border bg-card/90 backdrop-blur px-5 py-4 shadow-md"
    >
      <code className="truncate font-mono text-sm md:text-base text-foreground cursor-blink">
        $ {cmd}
      </code>
      <button
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy command"}
        className="inline-flex shrink-0 cursor-pointer items-center justify-center text-muted-foreground transition hover:text-foreground"
      >
        {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      </button>
    </motion.div>
  );
}

function DemoVideo() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.9, duration: 0.7 }}
      className="relative mt-10 block aspect-video w-full max-w-3xl overflow-hidden rounded-2xl border border-border bg-background shadow-xl"
    >
      <iframe
        src="https://www.youtube.com/embed/dQw4w9WgXcQ"
        title="Sensus demo"
        className="absolute inset-0 h-full w-full"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowFullScreen
      />
    </motion.div>
  );
}

function Hero() {
  return (
    <section className="relative min-h-screen w-full overflow-hidden bg-background">
      <div className="absolute inset-0">
        <SoundWaves />
      </div>
      <div className="relative z-10 flex min-h-screen items-center justify-center px-6 py-20">
        <div className="flex w-full flex-col items-center text-center">
          <motion.img
            src={logo}
            alt="Sensus"
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: 0.05, duration: 0.9 }}
            className="h-36 md:h-48 w-auto -mb-8 md:-mb-12 select-none"
            draggable={false}
          />
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.9 }}
            className="text-6xl md:text-7xl font-bold tracking-tight text-foreground"
          >
            Sensus
          </motion.h1>
          <InstallLine />
          <DemoVideo />
        </div>
      </div>
    </section>
  );
}

const STATS = [
  {
    value: "2.2B",
    label: "people live with a vision impairment.",
    source: "WHO",
  },
  {
    value: "1.3B",
    label: "live with a significant disability.",
    source: "WHO",
  },
  {
    value: "40%",
    label: "of adults 65+ struggle with computers.",
    source: "Pew Research",
  },
  {
    value: "90%",
    label: "of websites fail assistive technology.",
    source: "WebAIM",
  },
] as const;

function BuiltWith() {
  return (
    <section className="relative border-t border-border bg-card/40 py-28 px-6">
      <div className="mx-auto max-w-6xl">
        <div className="mb-20 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-foreground">
            Built for the people computers forgot.
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-16 gap-y-20">
          {STATS.map((s, i) => (
            <motion.div
              key={s.value}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{
                delay: i * 0.08,
                duration: 0.8,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="flex flex-col items-center text-center"
            >
              <div className="text-7xl md:text-8xl font-bold tracking-tight text-foreground leading-none">
                {s.value}
              </div>
              <p className="mt-5 text-base md:text-lg text-foreground/80 leading-snug whitespace-nowrap">
                {s.label}
              </p>
              <div className="mt-3 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                {s.source}
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-24 border-t border-border pt-12">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border rounded-xl overflow-hidden">
            {[
              { num: 3, title: "Good Health & Well-being" },
              { num: 8, title: "Decent Work & Economic Growth" },
              { num: 9, title: "Industry, Innovation & Infrastructure" },
              { num: 10, title: "Reduced Inequalities" },
            ].map((g) => (
              <div
                key={g.num}
                className="flex flex-col items-center justify-center gap-2 bg-card px-4 py-8 text-center"
              >
                <div className="font-mono text-xs text-muted-foreground">SDG</div>
                <div className="text-5xl font-bold tracking-tight text-foreground leading-none">
                  {g.num}
                </div>
                <div className="mt-2 text-sm text-foreground/80 leading-snug">
                  {g.title}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function IBMBadge() {
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-foreground px-1.5 py-0.5 font-mono text-[10px] font-bold tracking-wider text-background">
      IBM
    </span>
  );
}

type FlowNode = {
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  label: string;
  ibm?: boolean;
  size?: "lg" | "md";
};

function FlowStep({ node }: { node: FlowNode }) {
  const Icon = node.icon;
  const big = (node.size ?? "lg") === "lg";
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      className="flex flex-col items-center gap-3"
    >
      <div className="relative">
        <Icon
          className={`${big ? "h-16 w-16" : "h-10 w-10"} text-foreground`}
          strokeWidth={1.4}
        />
        {node.ibm && (
          <div className="absolute -top-2 -right-6">
            <IBMBadge />
          </div>
        )}
      </div>
      <h3
        className={`${
          big ? "text-xl md:text-2xl" : "text-sm md:text-base"
        } font-semibold tracking-tight text-foreground text-center`}
      >
        {node.label}
      </h3>
    </motion.div>
  );
}

function FlowArrow({ height = "h-12" }: { height?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, scaleY: 0 }}
      whileInView={{ opacity: 1, scaleY: 1 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="flex flex-col items-center origin-top"
      aria-hidden
    >
      <div className={`${height} w-px bg-border`} />
      <ArrowDown className="h-5 w-5 text-muted-foreground -mt-1" />
    </motion.div>
  );
}

function FlowFanOut({ children }: { children: FlowNode[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      className="w-full"
    >
      {/* Fan-out connector */}
      <div className="relative mx-auto h-10 w-full max-w-2xl" aria-hidden>
        <div className="absolute left-1/2 top-0 h-5 w-px -translate-x-1/2 bg-border" />
        <div
          className="absolute top-5 h-px bg-border"
          style={{
            left: `${100 / (children.length * 2)}%`,
            right: `${100 / (children.length * 2)}%`,
          }}
        />
        {children.map((_, i) => (
          <div
            key={i}
            className="absolute top-5 h-5 w-px bg-border"
            style={{ left: `${(100 / children.length) * (i + 0.5)}%` }}
          />
        ))}
      </div>
      <div
        className="grid gap-6"
        style={{ gridTemplateColumns: `repeat(${children.length}, minmax(0, 1fr))` }}
      >
        {children.map((c) => (
          <FlowStep key={c.label} node={{ ...c, size: "md" }} />
        ))}
      </div>
    </motion.div>
  );
}

type FlowItem = { node: FlowNode; children?: FlowNode[] };

function HowItWorks() {
  const flow: FlowItem[] = [
    { node: { icon: Mic, label: "Microphone" } },
    { node: { icon: Waves, label: "Watson Speech-to-Text", ibm: true } },
    {
      node: { icon: Compass, label: "Intent Router" },
      children: [
        { icon: Zap, label: "Fast Path" },
        { icon: Brain, label: "Full Orchestration" },
      ],
    },
    {
      node: { icon: Brain, label: "Watson AI Orchestrator", ibm: true },
      children: [
        { icon: LayoutGrid, label: "Chat" },
        { icon: Code2, label: "Tool Call" },
      ],
    },
    {
      node: { icon: LayoutGrid, label: "Specialized Agents" },
      children: [
        { icon: Globe, label: "Browser" },
        { icon: Monitor, label: "Desktop" },
        { icon: Terminal, label: "Shell" },
        { icon: Keyboard, label: "Shortcuts" },
        { icon: Code2, label: "Coding" },
      ],
    },
    { node: { icon: Database, label: "Db2 Memory", ibm: true } },
    { node: { icon: Volume2, label: "Watson Text-to-Speech", ibm: true } },
  ];

  return (
    <section className="relative border-t border-border bg-background py-32 px-6">
      <div className="mx-auto max-w-4xl">
        <div className="mb-20 text-center">
          <h2 className="text-3xl md:text-5xl font-bold text-foreground whitespace-nowrap">
            How Sensus hears, thinks, and acts.
          </h2>
        </div>

        <div className="flex flex-col items-center">
          {flow.map((item, i) => (
            <div key={item.node.label} className="flex flex-col items-center w-full">
              <FlowStep node={item.node} />
              {item.children && (
                <>
                  <FlowFanOut children={item.children} />
                </>
              )}
              {i < flow.length - 1 && <FlowArrow />}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Index() {
  return (
    <main className="bg-background">
      <Hero />
      <HowItWorks />
      <BuiltWith />
    </main>
  );
}
