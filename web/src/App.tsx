import { useCallback, useMemo, useState } from "react"
import { imageForCategory } from "@/lib/productImages"
import { ChevronDown, HelpCircle, RefreshCw, Send, ShoppingBag, Sparkles } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"
import {
  SCENARIOS,
  type BatchRow,
  type ScenarioName,
  type SimState,
  defaultBriefTemplate,
  simReset,
  simStep,
} from "@/lib/api"

/** Plain-language labels for non-technical users */
const STORE_MODES: Record<
  ScenarioName,
  { title: string; subtitle: string }
> = {
  STABLE_WEEK: { title: "Typical week", subtitle: "Steady demand — good for learning the shelf." },
  BUSY_WEEKEND: { title: "Busy weekend", subtitle: "More orders and a trend or two to watch." },
  FARMER_WEEK: { title: "Farmer offers week", subtitle: "Extra bulk deals from suppliers." },
  TREND_WEEK: { title: "Trending & events", subtitle: "Social buzz plus a busier day in the mix." },
  CRISIS_WEEK: { title: "High-pressure week", subtitle: "Everything at once — stress test." },
}

function urgencyLabel(u: string): { text: string; className: string } {
  const x = u.toUpperCase()
  if (x === "CRITICAL") return { text: "Clear today", className: "bg-rose-100 text-rose-800 border-rose-200" }
  if (x === "URGENT") return { text: "Discount soon", className: "bg-amber-100 text-amber-900 border-amber-200" }
  if (x === "WATCH") return { text: "Watch shelf", className: "bg-sky-100 text-sky-900 border-sky-200" }
  return { text: "Fresh", className: "bg-emerald-50 text-emerald-900 border-emerald-200" }
}

function ProductCard({ b }: { b: BatchRow }) {
  const pill = urgencyLabel(b.urgency)
  const initial = (b.category || "?").slice(0, 1).toUpperCase()
  const [imgFailed, setImgFailed] = useState(false)
  const src = imageForCategory(b.category)
  const prev = b.previous_price
  const priceMoved = prev !== undefined && Math.abs(prev - b.current_price) > 0.005
  const down = priceMoved && prev !== undefined && b.current_price < prev

  return (
    <article className="flex flex-col overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm">
      <div className="relative h-40 overflow-hidden bg-gradient-to-br from-zinc-100 to-zinc-50">
        {!imgFailed ? (
          <img
            src={src}
            alt=""
            className="h-full w-full object-cover"
            loading="lazy"
            referrerPolicy="no-referrer-when-downgrade"
            onError={() => setImgFailed(true)}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <span className="text-5xl font-semibold text-zinc-300">{initial}</span>
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-2 p-4">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h3 className="font-semibold leading-tight text-zinc-900">{b.category}</h3>
            <p className="text-xs text-zinc-500">SKU {b.batch_id}</p>
          </div>
          <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${pill.className}`}>{pill.text}</span>
        </div>
        {priceMoved ? (
          <p className={`text-xs font-medium ${down ? "text-emerald-700" : "text-amber-800"}`}>
            {down ? "Price dropped" : "Price updated"}: ₹{prev} → ₹{b.current_price}
          </p>
        ) : null}
        <div className="mt-auto flex items-end justify-between gap-2">
          <div>
            <p className="text-2xl font-bold tracking-tight text-zinc-900">₹{b.current_price}</p>
            <p className="text-sm text-zinc-400 line-through">list ₹{b.original_price}</p>
          </div>
          <div className="text-right text-sm">
            <p className="font-medium text-emerald-700">−{b.discount_pct}% vs list</p>
            <p className="text-xs text-zinc-500">{b.hours_to_expiry}h · qty {b.quantity}</p>
          </div>
        </div>
      </div>
    </article>
  )
}

export default function App() {
  const [phase, setPhase] = useState<"pick" | "store">("pick")
  const [scenario, setScenario] = useState<ScenarioName>("STABLE_WEEK")
  const [seed] = useState(() => Math.floor(Math.random() * 10_000))
  const [brief, setBrief] = useState("")
  const [observation, setObservation] = useState("")
  const [state, setState] = useState<SimState | null>(null)
  const [batches, setBatches] = useState<BatchRow[]>([])
  const [wrrHistory, setWrrHistory] = useState<number[]>([])
  const [stepCount, setStepCount] = useState(0)
  const [done, setDone] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [parseOk, setParseOk] = useState(true)
  /** Latest server-built brief (real batch / offer ids). */
  const [suggestedHint, setSuggestedHint] = useState("")

  const engine = state?.engine_type ?? "PRICING"
  const mode = STORE_MODES[scenario]

  const openStore = useCallback(async () => {
    setLoading(true)
    setError(null)
    setDone(false)
    try {
      const r = await simReset(scenario, seed)
      setObservation(r.observation)
      setState(r.state)
      setBatches(r.batches)
      setWrrHistory(r.wrr_history)
      setStepCount(r.step_count)
      const hint = r.suggested_brief?.trim() ?? ""
      setSuggestedHint(hint)
      setBrief(
        hint ||
          defaultBriefTemplate((r.engine_type ?? r.state?.engine_type ?? "PRICING") as string),
      )
      setParseOk(true)
      setPhase("store")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [scenario, seed])

  const applySuggestedBrief = useCallback(() => {
    setBrief(suggestedHint.trim() || defaultBriefTemplate(engine))
  }, [engine, suggestedHint])

  const runUpdate = useCallback(async () => {
    if (done || !brief.trim()) return
    setLoading(true)
    setError(null)
    try {
      const r = await simStep(brief)
      setObservation(r.observation)
      setState(r.state)
      setBatches(r.batches)
      setWrrHistory(r.wrr_history)
      setStepCount(r.step_count)
      setParseOk(r.parse_success !== false)
      const nextHint = r.suggested_brief?.trim() ?? ""
      setSuggestedHint(nextHint)
      if (r.done) {
        setDone(true)
        setBrief("")
      } else {
        setBrief(nextHint || defaultBriefTemplate((r.next_engine_type ?? "PRICING") as string))
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [brief, done])

  const wrrNow = state?.wrr_so_far
  const shelfTitle = useMemo(
    () => (phase === "store" ? `${mode.title} · live prices` : "Choose how your week looks"),
    [phase, mode.title],
  )

  if (phase === "pick") {
    return (
      <div className="min-h-screen bg-zinc-50 text-zinc-900">
        <header className="border-b border-zinc-200 bg-white">
          <div className="mx-auto flex max-w-3xl items-center gap-3 px-4 py-5">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-600 text-white shadow-md">
              <ShoppingBag className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">FreshQuick</h1>
              <p className="text-sm text-zinc-500">Demo store — pick a week, then see your shelf.</p>
            </div>
          </div>
        </header>

        <main className="mx-auto max-w-3xl px-4 py-10">
          {error ? (
            <Card className="mb-6 border-red-200 bg-red-50">
              <CardHeader>
                <CardTitle className="text-red-900 text-base">Can&apos;t reach the store backend</CardTitle>
                <CardDescription className="text-red-800">
                  Run <code className="rounded bg-white px-1.5 py-0.5 text-zinc-800">python -m uvicorn server.app:app --port 8000</code> then refresh.{" "}
                  <span className="block mt-1 text-sm">{error}</span>
                </CardDescription>
              </CardHeader>
            </Card>
          ) : null}

          <p className="mb-6 text-center text-lg text-zinc-600">How busy is this week for your store?</p>

          <div className="grid gap-3 sm:grid-cols-2">
            {SCENARIOS.map((s) => {
              const m = STORE_MODES[s]
              const active = scenario === s
              return (
                <button
                  key={s}
                  type="button"
                  onClick={() => setScenario(s)}
                  className={`rounded-2xl border-2 p-5 text-left transition-all ${
                    active
                      ? "border-emerald-600 bg-emerald-50/80 shadow-md ring-2 ring-emerald-600/20"
                      : "border-zinc-200 bg-white hover:border-zinc-300"
                  }`}
                >
                  <p className="font-semibold text-zinc-900">{m.title}</p>
                  <p className="mt-1 text-sm text-zinc-600">{m.subtitle}</p>
                </button>
              )
            })}
          </div>

          <Button className="mx-auto mt-10 flex h-12 w-full max-w-md text-base" size="lg" onClick={openStore} disabled={loading}>
            {loading ? <RefreshCw className="h-5 w-5 animate-spin" /> : <ShoppingBag className="h-5 w-5" />}
            Open store &amp; load prices
          </Button>

          <details className="mx-auto mt-12 max-w-lg rounded-xl border border-zinc-200 bg-white p-4 text-sm text-zinc-600">
            <summary className="cursor-pointer font-medium text-zinc-800">What is this demo?</summary>
            <p className="mt-2">
              Behind the scenes this is the QStorePrice simulation: each time you tap <strong>Update prices for this window</strong>,
              the store runs one pricing cycle. Technical terms (WRR, briefs) stay optional below when you&apos;re in the store.
            </p>
          </details>
        </main>
      </div>
    )
  }

  /* --- Store view --- */
  return (
    <div className="min-h-screen bg-zinc-100 text-zinc-900">
      <header className="sticky top-0 z-10 border-b border-zinc-200 bg-white/95 shadow-sm backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-white">
              <ShoppingBag className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs font-medium uppercase tracking-wide text-emerald-700">FreshQuick</p>
              <p className="font-semibold text-zinc-900">{mode.title}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {typeof wrrNow === "number" ? (
              <div className="text-right text-xs sm:text-sm">
                <Badge variant="secondary" className="border border-zinc-200 bg-zinc-50 text-zinc-800">
                  Waste recovery <span className="font-mono font-semibold">{wrrNow.toFixed(3)}</span>
                </Badge>
                {wrrHistory.length > 1 ? (
                  <p className="mt-1 text-[11px] text-zinc-500">
                    Recent: {wrrHistory.slice(-5).map((x) => x.toFixed(2)).join(" → ")}
                  </p>
                ) : null}
              </div>
            ) : null}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setPhase("pick")
                setObservation("")
                setState(null)
                setBatches([])
                setWrrHistory([])
                setDone(false)
              }}
            >
              Change week
            </Button>
            <Button variant="secondary" size="sm" onClick={openStore} disabled={loading}>
              <RefreshCw className={loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} />
              Reload shelf
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {error ? (
          <Card className="mb-4 border-red-200 bg-red-50">
            <CardContent className="py-4 text-sm text-red-900">{error}</CardContent>
          </Card>
        ) : null}

        <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">{shelfTitle}</h2>
            <p className="text-sm text-zinc-500">
              {batches.length} items on shelf · step {stepCount}
              {!parseOk ? <span className="text-amber-700"> · last action used a safe fallback</span> : null}
            </p>
          </div>
        </div>

        {batches.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="py-16 text-center text-zinc-500">No products loaded. Tap Reload shelf.</CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {batches.map((b) => (
              <ProductCard key={b.batch_id} b={b} />
            ))}
          </div>
        )}

        <Card className="mt-8 border-zinc-200 bg-white shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Store actions</CardTitle>
            <CardDescription>
              Each tap runs one store decision window. Prices update when the note uses your real SKU codes (we fill those
              automatically). Product photos load from the internet (Unsplash).
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
            <Button size="lg" className="gap-2" onClick={runUpdate} disabled={done || loading || !brief.trim()}>
              <Send className="h-4 w-4" />
              {done ? "Day finished" : "Update prices for this window"}
            </Button>
            <Button type="button" variant="outline" size="lg" className="gap-2" onClick={applySuggestedBrief} disabled={done || loading}>
              <Sparkles className="h-4 w-4" />
              Fill suggested manager note
            </Button>
            {done ? <Badge variant="warn">Week complete — use Change week or Reload shelf</Badge> : null}
          </CardContent>
        </Card>

        <details className="mt-6 rounded-xl border border-zinc-200 bg-white">
          <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 font-medium text-zinc-800">
            <ChevronDown className="h-4 w-4 shrink-0 opacity-60" />
            Advanced: manager note &amp; system readout
          </summary>
          <div className="border-t border-zinc-100 px-4 py-4">
            <p className="mb-2 flex items-center gap-1 text-xs text-zinc-500">
              <HelpCircle className="h-3.5 w-3.5" />
              Current focus: <strong>{engine}</strong> — the note below is what the simulator reads (includes a small JSON block).
            </p>
            <Textarea value={brief} onChange={(e) => setBrief(e.target.value)} disabled={done || loading} rows={8} className="mb-3" />
            <Separator className="my-3" />
            <p className="mb-1 text-xs font-medium text-zinc-500">Full system prompt (optional read)</p>
            <ScrollArea className="h-48 rounded-md border border-zinc-200 bg-zinc-50">
              <pre className="whitespace-pre-wrap p-3 text-[11px] leading-relaxed text-zinc-600">{observation || "—"}</pre>
            </ScrollArea>
          </div>
        </details>
      </main>
    </div>
  )
}
