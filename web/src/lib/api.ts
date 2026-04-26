export const SCENARIOS = [
  "STABLE_WEEK",
  "BUSY_WEEKEND",
  "FARMER_WEEK",
  "TREND_WEEK",
  "CRISIS_WEEK",
] as const

export type ScenarioName = (typeof SCENARIOS)[number]

export interface SimState {
  tick?: number
  day_of_week?: number
  hour_of_day?: number
  scenario?: string
  wrr_so_far?: number
  active_batches?: number
  critical_batches?: number
  pending_offers?: number
  active_trends?: number
  risk_buffer_balance?: number
  engine_type?: string
  episode_complete?: boolean
  weather?: string
  event?: string
  status?: string
}

export interface BatchRow {
  batch_id: string
  category: string
  urgency: string
  current_price: number
  original_price: number
  discount_pct: number
  hours_to_expiry: number
  quantity: number
  /** Shelf price before the last step (only if it changed). */
  previous_price?: number
}

export interface StepResponse {
  ok: boolean
  observation: string
  reward?: number
  done: boolean
  state: SimState
  batches: BatchRow[]
  step_count: number
  wrr_history: number[]
  engine_type?: string
  parse_success?: boolean
  quality_score?: number
  next_engine_type?: string | null
  final_reward?: Record<string, unknown>
  /** Server-built brief using real batch/offer ids — use so prices actually update. */
  suggested_brief?: string
}

async function parseJson(res: Response): Promise<unknown> {
  const text = await res.text()
  try {
    return JSON.parse(text)
  } catch {
    throw new Error(text.slice(0, 200) || `HTTP ${res.status}`)
  }
}

function errMessage(data: unknown, fallback: string): string {
  const d = data as { detail?: unknown }
  if (typeof d.detail === "string") return d.detail
  if (Array.isArray(d.detail) && d.detail[0]?.msg) return String(d.detail[0].msg)
  return fallback
}

export async function simReset(scenario: string, seed: number): Promise<StepResponse> {
  const res = await fetch("/api/sim/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario, seed }),
  })
  const data = await parseJson(res)
  if (!res.ok) throw new Error(errMessage(data, res.statusText))
  return data as StepResponse
}

export async function simStep(briefText: string): Promise<StepResponse> {
  const res = await fetch("/api/sim/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ brief_text: briefText }),
  })
  const data = await parseJson(res)
  if (!res.ok) throw new Error(errMessage(data, res.statusText))
  return data as StepResponse
}

export function defaultBriefTemplate(engine: string): string {
  const e = engine.toUpperCase()
  if (e === "FARMER") {
    return `SITUATION: [Farmer offer context]
SIGNAL ANALYSIS: N/A
VIABILITY CHECK: Shelf Life: PASS. Conflict: PASS.
RECOMMENDATION: [Accept / Counter / Decline]
DIRECTIVE: {"engine": "FARMER", "actions": [{"offer_id": "F000A", "decision": "ACCEPT", "counter_price": null}]}
CONFIDENCE: MEDIUM`
  }
  if (e === "TREND") {
    return `SITUATION: [Trend signal]
SIGNAL ANALYSIS: [Strength]
VIABILITY CHECK: Recipe Simplicity: PASS.
RECOMMENDATION: [Approve / Decline]
DIRECTIVE: {"engine": "TREND", "actions": [{"category": "PRODUCE", "decision": "APPROVE", "order_quantity_kg": 10.0}]}
CONFIDENCE: MEDIUM`
  }
  return `SITUATION: [Inventory & urgency]
SIGNAL ANALYSIS: N/A
VIABILITY CHECK: N/A
RECOMMENDATION: [Pricing strategy]
DIRECTIVE: {"engine": "PRICING", "actions": [{"batch_id": "B001", "price_multiplier": 0.85, "flash_sale": false, "bundle_with": null}]}
CONFIDENCE: MEDIUM`
}
