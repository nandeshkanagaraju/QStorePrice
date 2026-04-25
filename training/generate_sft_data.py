"""Generate SFT warm-start training data for all three engines.

Writes three JSON files to training/sft_data/:
  pricing_examples.json  — PRICING engine, 10 easy + 10 medium + 10 hard
  farmer_examples.json   — FARMER engine, 10 easy + 10 medium + 10 hard
  trend_examples.json    — TREND engine, 10 easy + 10 medium + 10 hard

Each example:  {"engine_type", "difficulty", "prompt", "completion"}
Completion must contain all 6 section headers required by the brief pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URGENCY_EMOJI = {
    "FRESH": "🟢",
    "WATCH": "🟡",
    "URGENT": "🟠",
    "CRITICAL": "🔴",
}

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_CATEGORIES = ["fruits", "vegetables", "dairy", "mushrooms", "leafy_greens", "herbs"]
_FARMER_NAMES = [
    "Ramesh Farms", "Kavitha Organics", "Suresh Agro", "Priya Fresh Produce",
    "Lakshmi Dairy Co.", "Gopal Vegetables", "Meena Herbs", "Kumar Fruits",
]


def _hour_str(h: int) -> str:
    if h == 0:
        return "12:00 AM"
    if h < 12:
        return f"{h}:00 AM"
    if h == 12:
        return "12:00 PM"
    return f"{h - 12}:00 PM"


# ---------------------------------------------------------------------------
# PRICING examples
# ---------------------------------------------------------------------------

def _pricing_prompt(batches: list[dict], day: int, hour: int, buffer: float) -> str:
    lines = ["=== CURRENT INVENTORY ==="]
    for b in batches:
        emoji = _URGENCY_EMOJI[b["urgency"]]
        lines.append(f"[{emoji}] {b['category']} — Batch {b['batch_id']}")
        lines.append(
            f"  Stock: {b['qty']} units | Expiry: {b['hrs']:.1f}hrs | Urgency: {b['urgency']}"
        )
        lines.append(
            f"  Current price: Rs {b['price']:.0f} | Original: Rs {b['orig']:.0f} | Floor: Rs {b['floor']:.0f}"
        )
        lines.append(f"  Velocity: {b['vel']:.2f} units/hr | Discount: {b['disc']:.0f}%")
        lines.append("")
    lines.append("=== MARKET CONTEXT ===")
    lines.append(
        f"Day: {_DAY_NAMES[day]} | Hour: {_hour_str(hour)} | Risk Buffer: Rs {buffer:.0f}"
    )
    credits = " | ".join(f"{c}: 3" for c in set(b["category"] for b in batches))
    lines.append(f"Notification credits remaining: {credits}")
    lines.append("")
    lines.append("=== YOUR TASK ===")
    lines.append(
        "Write a PRICING Operating Brief. For each URGENT or CRITICAL batch, "
        "decide the appropriate price_multiplier. For FRESH and WATCH batches, "
        "you may hold price (1.0) or apply a small early discount."
    )
    lines.append(
        "Remember: price_multiplier below 0.35 on batches with > 48hrs remaining "
        "triggers an anti-hack penalty."
    )
    return "\n".join(lines)


def _pricing_completion(batches: list[dict]) -> str:
    urgent = [b for b in batches if b["urgency"] in ("URGENT", "CRITICAL")]
    fresh = [b for b in batches if b["urgency"] in ("FRESH", "WATCH")]

    if urgent:
        situation = (
            f"{len(urgent)} batch(es) require immediate attention: "
            + ", ".join(f"{b['batch_id']} ({b['category']}, {b['hrs']:.1f}hrs)" for b in urgent)
            + f". {len(fresh)} batch(es) are stable."
        )
    else:
        situation = (
            f"All {len(batches)} batch(es) are in FRESH or WATCH urgency. "
            "No immediate clearance needed."
        )

    # Build actions
    actions = []
    for b in batches:
        if b["urgency"] == "CRITICAL":
            mult = round(max(b["floor"] / b["orig"], 0.35), 2)
            flash = b["hrs"] < 3
        elif b["urgency"] == "URGENT":
            mult = round(max(b["floor"] / b["orig"] + 0.10, 0.50), 2)
            flash = False
        elif b["urgency"] == "WATCH":
            mult = 1.0
            flash = False
        else:
            mult = 1.0
            flash = False
        mult = min(mult, 1.0)
        actions.append({
            "batch_id": b["batch_id"],
            "price_multiplier": mult,
            "flash_sale": flash,
            "bundle_with": None,
        })

    if urgent:
        recommendation = (
            f"Apply targeted discounts to {len(urgent)} at-risk batch(es) to drive clearance "
            "before expiry. Hold prices on stable stock."
        )
        confidence = "HIGH" if len(urgent) <= 2 else "MEDIUM"
    else:
        recommendation = "Maintain current pricing. No clearance discounts required this cycle."
        confidence = "HIGH"

    directive = json.dumps({"engine": "PRICING", "actions": actions}, separators=(", ", ": "))

    return (
        f"SITUATION: {situation}\n\n"
        "SIGNAL ANALYSIS: N/A\n\n"
        "VIABILITY CHECK: N/A\n\n"
        f"RECOMMENDATION: {recommendation}\n\n"
        f"DIRECTIVE:\n{directive}\n\n"
        f"CONFIDENCE: {confidence}"
    )


def _make_batch(batch_id: str, urgency: str, category: str | None = None, rng: random.Random | None = None) -> dict:
    if rng is None:
        rng = random.Random(42)
    cat = category or rng.choice(_CATEGORIES)
    prices = {
        "fruits": (65, 28), "vegetables": (38, 16), "dairy": (70, 30),
        "mushrooms": (38, 16), "leafy_greens": (25, 10), "herbs": (25, 10),
    }
    orig, floor_base = prices.get(cat, (50, 20))
    floor = floor_base + rng.randint(0, 5)
    if urgency == "FRESH":
        hrs = rng.uniform(72, 120)
    elif urgency == "WATCH":
        hrs = rng.uniform(24, 72)
    elif urgency == "URGENT":
        hrs = rng.uniform(6, 24)
    else:
        hrs = rng.uniform(1, 6)
    disc = 0 if urgency in ("FRESH", "WATCH") else rng.randint(10, 30)
    price = orig * (1 - disc / 100)
    vel_base = {"fruits": 3.0, "vegetables": 2.5, "dairy": 4.0, "mushrooms": 1.5, "leafy_greens": 2.0, "herbs": 1.0}
    vel = vel_base.get(cat, 2.0) * rng.uniform(0.5, 1.5)
    qty = int(vel * hrs * rng.uniform(0.5, 1.5))
    return {
        "batch_id": batch_id, "category": cat, "urgency": urgency,
        "hrs": hrs, "qty": max(qty, 1), "price": price, "orig": orig,
        "floor": floor, "vel": vel, "disc": disc,
    }


def generate_pricing_examples(n_per_difficulty: int = 10) -> list[dict]:
    examples = []
    rng = random.Random(1001)

    # EASY: 1-2 batches, mostly URGENT/CRITICAL
    for i in range(n_per_difficulty):
        batches = [
            _make_batch(f"P{i:03d}A", "CRITICAL", rng=rng),
            _make_batch(f"P{i:03d}B", "FRESH", rng=rng),
        ]
        prompt = _pricing_prompt(batches, rng.randint(0, 6), rng.randint(8, 20), 5000)
        examples.append({"engine_type": "PRICING", "difficulty": "easy",
                         "prompt": prompt, "completion": _pricing_completion(batches)})

    # MEDIUM: 3-4 batches, mixed urgencies
    for i in range(n_per_difficulty):
        batches = [
            _make_batch(f"M{i:03d}A", "URGENT", rng=rng),
            _make_batch(f"M{i:03d}B", "WATCH", rng=rng),
            _make_batch(f"M{i:03d}C", "FRESH", rng=rng),
        ]
        prompt = _pricing_prompt(batches, rng.randint(0, 6), rng.randint(8, 20), rng.uniform(2000, 5000))
        examples.append({"engine_type": "PRICING", "difficulty": "medium",
                         "prompt": prompt, "completion": _pricing_completion(batches)})

    # HARD: 4-5 batches, multiple CRITICAL, low buffer
    for i in range(n_per_difficulty):
        batches = [
            _make_batch(f"H{i:03d}A", "CRITICAL", rng=rng),
            _make_batch(f"H{i:03d}B", "CRITICAL", rng=rng),
            _make_batch(f"H{i:03d}C", "URGENT", rng=rng),
            _make_batch(f"H{i:03d}D", "WATCH", rng=rng),
        ]
        prompt = _pricing_prompt(batches, rng.randint(0, 6), rng.randint(8, 20), rng.uniform(500, 1500))
        examples.append({"engine_type": "PRICING", "difficulty": "hard",
                         "prompt": prompt, "completion": _pricing_completion(batches)})

    return examples


# ---------------------------------------------------------------------------
# FARMER examples
# ---------------------------------------------------------------------------

def _farmer_prompt(offers: list[dict], inv_batches: list[dict], buffer: float) -> str:
    lines = ["=== PENDING FARMER OFFERS ==="]
    for o in offers:
        viab = o["viability"]
        if viab >= 0.80:
            label = "Strong Accept"
        elif viab >= 0.60:
            label = "Acceptable"
        elif viab >= 0.40:
            label = "Borderline"
        else:
            label = "High Risk"
        lines.append(f"Offer {o['offer_id']}: {o['farmer']} — {o['qty_kg']:.0f}kg of {o['product']}")
        lines.append(f"  Offered price: Rs {o['price_kg']:.0f}/kg")
        lines.append(f"  Seller-reported shelf life: {o['shelf_life_hrs']}hrs")
        lines.append(f"  Viability score: {viab:.2f} ({label})")
        lines.append("")
    lines.append("=== CURRENT INVENTORY (same category) ===")
    if inv_batches:
        for b in inv_batches:
            emoji = _URGENCY_EMOJI[b["urgency"]]
            lines.append(
                f"[{emoji}] {b['category']} — Batch {b['batch_id']} | "
                f"{b['qty']} units | {b['hrs']:.1f}hrs | Rs {b['price']:.0f} | Vel: {b['vel']:.2f}/hr"
            )
    else:
        lines.append("No active inventory in offered categories.")
    lines.append("")
    lines.append("=== RISK BUFFER ===")
    lines.append(f"Current balance: Rs {buffer:.0f}")
    lines.append("Note: Buffer below Rs 2000 means conservative acceptance only.")
    if buffer < 2000:
        lines.append("⚠️ Low buffer — only accept offers with viability >= 0.80")
    lines.append("")
    lines.append("=== YOUR TASK ===")
    lines.append(
        "Write a FARMER Operating Brief. For each pending offer, decide ACCEPT, COUNTER, "
        "or DECLINE. If countering, specify a counter_price in the DIRECTIVE."
    )
    lines.append(
        "Remember: accepting an offer with viability_score < 0.30 triggers an anti-hack penalty (reckless acceptance)."
    )
    return "\n".join(lines)


def _farmer_completion(offers: list[dict], buffer: float) -> str:
    low_buffer = buffer < 2000

    offer_summaries = []
    actions = []
    viability_lines = []

    for o in offers:
        viab = o["viability"]
        viab_label = "Strong Accept" if viab >= 0.80 else "Acceptable" if viab >= 0.60 else "Borderline" if viab >= 0.40 else "High Risk"
        factor = "PASS" if viab >= 0.60 else ("FLAG" if viab >= 0.40 else "FAIL")
        viability_lines.append(f"{o['offer_id']}: viability {viab:.2f} — {factor} ({viab_label})")

        if viab >= 0.80 and not low_buffer:
            decision = "ACCEPT"
            counter = None
            offer_summaries.append(f"{o['offer_id']} (viab={viab:.2f}, strong accept)")
        elif viab >= 0.60 and not low_buffer:
            decision = "ACCEPT"
            counter = None
            offer_summaries.append(f"{o['offer_id']} (viab={viab:.2f}, acceptable)")
        elif viab >= 0.40 and not low_buffer:
            # Counter at 10% below offered
            counter = round(o["price_kg"] * 0.90, 1)
            decision = "COUNTER"
            offer_summaries.append(f"{o['offer_id']} (viab={viab:.2f}, countering at Rs {counter})")
        else:
            decision = "DECLINE"
            counter = None
            offer_summaries.append(f"{o['offer_id']} (viab={viab:.2f}, declining)")

        actions.append({"offer_id": o["offer_id"], "decision": decision, "counter_price": counter})

    accepted = [a for a in actions if a["decision"] == "ACCEPT"]
    declined = [a for a in actions if a["decision"] == "DECLINE"]
    countered = [a for a in actions if a["decision"] == "COUNTER"]

    situation = (
        f"{len(offers)} farmer offer(s) pending evaluation. "
        + (f"Risk buffer is {'LOW (Rs {:.0f})'.format(buffer)} — conservative mode active. " if low_buffer else "")
        + f"Decided: {len(accepted)} ACCEPT, {len(countered)} COUNTER, {len(declined)} DECLINE."
    )

    if accepted:
        recommendation = f"Accept {len(accepted)} viable offer(s) to replenish stock at competitive cost."
    elif countered:
        recommendation = f"Counter {len(countered)} borderline offer(s) to reduce procurement cost."
    else:
        recommendation = "Decline all offers — viability scores too low or buffer too constrained for risk."

    confidence = "HIGH" if all(o["viability"] > 0.60 or o["viability"] < 0.30 for o in offers) else "MEDIUM"
    directive = json.dumps({"engine": "FARMER", "actions": actions}, separators=(", ", ": "))

    return (
        f"SITUATION: {situation}\n\n"
        f"SIGNAL ANALYSIS: N/A\n\n"
        f"VIABILITY CHECK: {' | '.join(viability_lines)}\n\n"
        f"RECOMMENDATION: {recommendation}\n\n"
        f"DIRECTIVE:\n{directive}\n\n"
        f"CONFIDENCE: {confidence}"
    )


def _make_offer(offer_id: str, viability: float, rng: random.Random) -> dict:
    cat = rng.choice(["vegetables", "fruits", "dairy", "herbs"])
    product = f"{cat}_{offer_id[-2:]}"
    prices = {"fruits": 40, "vegetables": 22, "dairy": 45, "herbs": 18}
    price_kg = prices.get(cat, 30) * rng.uniform(0.8, 1.3)
    qty_kg = rng.uniform(50, 200)
    shelf_hrs = int(rng.uniform(48, 120))
    return {
        "offer_id": offer_id, "farmer": rng.choice(_FARMER_NAMES),
        "product": product, "qty_kg": qty_kg, "price_kg": round(price_kg, 1),
        "shelf_life_hrs": shelf_hrs, "viability": viability, "category": cat,
    }


def generate_farmer_examples(n_per_difficulty: int = 10) -> list[dict]:
    examples = []
    rng = random.Random(2002)

    # EASY: 1 offer, clear viability
    for i in range(n_per_difficulty):
        viab = rng.uniform(0.75, 0.95)
        offers = [_make_offer(f"F{i:03d}A", viab, rng)]
        inv = [_make_batch(f"INV{i}A", rng.choice(["FRESH", "WATCH"]),
                           category=offers[0]["category"], rng=rng)]
        buffer = rng.uniform(4000, 6000)
        prompt = _farmer_prompt(offers, inv, buffer)
        examples.append({"engine_type": "FARMER", "difficulty": "easy",
                         "prompt": prompt, "completion": _farmer_completion(offers, buffer)})

    # MEDIUM: 2 offers, mixed viability
    for i in range(n_per_difficulty):
        offers = [
            _make_offer(f"G{i:03d}A", rng.uniform(0.65, 0.85), rng),
            _make_offer(f"G{i:03d}B", rng.uniform(0.35, 0.55), rng),
        ]
        inv = [_make_batch(f"INVM{i}A", "WATCH", category=offers[0]["category"], rng=rng)]
        buffer = rng.uniform(2500, 4000)
        prompt = _farmer_prompt(offers, inv, buffer)
        examples.append({"engine_type": "FARMER", "difficulty": "medium",
                         "prompt": prompt, "completion": _farmer_completion(offers, buffer)})

    # HARD: 3 offers, low buffer, borderline viability
    for i in range(n_per_difficulty):
        offers = [
            _make_offer(f"K{i:03d}A", rng.uniform(0.75, 0.90), rng),
            _make_offer(f"K{i:03d}B", rng.uniform(0.40, 0.60), rng),
            _make_offer(f"K{i:03d}C", rng.uniform(0.20, 0.35), rng),
        ]
        inv = [
            _make_batch(f"INVH{i}A", "URGENT", category=offers[0]["category"], rng=rng),
            _make_batch(f"INVH{i}B", "WATCH", category=offers[1]["category"], rng=rng),
        ]
        buffer = rng.uniform(800, 1800)
        prompt = _farmer_prompt(offers, inv, buffer)
        examples.append({"engine_type": "FARMER", "difficulty": "hard",
                         "prompt": prompt, "completion": _farmer_completion(offers, buffer)})

    return examples


# ---------------------------------------------------------------------------
# TREND examples
# ---------------------------------------------------------------------------

_SIGNAL_SOURCES = ["INSTAGRAM", "GOOGLE_TRENDS", "ZOMATO", "YOUTUBE"]


def _trend_prompt(signals: list[dict], inv_batches: list[dict], tick: int) -> str:
    lines = ["=== ACTIVE TREND SIGNALS ==="]
    for s in signals:
        tier = "HIGH" if s["score"] >= 80 else "MEDIUM" if s["score"] >= 65 else "LOW"
        lines.append(f"📱 {s['category'].upper()} — Score: {s['score']:.0f}/100 ({s['source']})")
        lines.append(f"  Suggested restock: {s['suggested_kg']:.0f}kg")
        lines.append("  Signal factors:")
        lines.append(f"    Recipe simplicity:     {s['recipe_simplicity']:.2f}")
        lines.append(f"    Ingredient rarity:     {s['ingredient_rarity']:.2f}")
        lines.append(f"    View velocity:         {s['view_velocity']:.2f}")
        lines.append(f"    Local relevance:       {s['local_relevance']:.2f}")
        lines.append(f"    Historical conversion: {s['historical_conversion']:.2f}")
        lines.append(f"  Confidence tier: {tier}")
        lines.append("")
    lines.append("=== CURRENT STOCK (affected categories) ===")
    if inv_batches:
        for b in inv_batches:
            emoji = _URGENCY_EMOJI[b["urgency"]]
            lines.append(
                f"[{emoji}] {b['category']} — Batch {b['batch_id']} | "
                f"{b['qty']} units | {b['hrs']:.1f}hrs | Rs {b['price']:.0f}"
            )
    else:
        lines.append("No active inventory in signalled categories.")
    lines.append("")
    lines.append("=== COOLDOWN STATUS ===")
    for s in signals:
        hours_ago = (tick - s["detected_at"]) / 4.0
        lines.append(f"  {s['category']}: signal detected {hours_ago:.1f}hrs ago")
    lines.append("")
    lines.append("=== YOUR TASK ===")
    lines.append(
        "Write a TREND Operating Brief. For each signal above threshold (65), "
        "decide APPROVE or DECLINE. If approving, specify order_quantity_kg."
    )
    lines.append("Cap: max order = avg weekly velocity x 2.0. Hard cap enforced by the engine.")
    return "\n".join(lines)


def _trend_completion(signals: list[dict]) -> str:
    actions = []
    viability_lines = []
    approved = []
    declined = []

    for s in signals:
        score = s["score"]
        if score >= 80:
            factor = "PASS"
            decision = "APPROVE"
            order_kg = round(s["suggested_kg"] * 0.85, 1)
            approved.append(s["category"])
        elif score >= 65:
            factor = "PASS"
            decision = "APPROVE"
            order_kg = round(s["suggested_kg"] * 0.60, 1)
            approved.append(s["category"])
        else:
            factor = "FAIL"
            decision = "DECLINE"
            order_kg = None
            declined.append(s["category"])

        viability_lines.append(
            f"{s['category']}: score {score:.0f}/100 — {factor} "
            f"({'above' if score >= 65 else 'below'} threshold of 65)"
        )
        actions.append({"category": s["category"], "decision": decision, "order_quantity_kg": order_kg})

    tier_desc = "high-confidence" if any(s["score"] >= 80 for s in signals) else "moderate"
    situation = (
        f"{len(signals)} trend signal(s) detected. "
        + (f"{len(approved)} signal(s) exceed threshold and warrant restocking. " if approved else "")
        + (f"{len(declined)} signal(s) below threshold — declining. " if declined else "")
        + f"All signals are {tier_desc} based on composite scores."
    )

    signal_analysis = "; ".join(
        f"{s['category']} trending via {s['source']} (score {s['score']:.0f}, "
        f"recipe_simplicity={s['recipe_simplicity']:.2f})"
        for s in signals
    )

    if approved:
        recommendation = (
            f"Approve restock for {', '.join(approved)} — signals are above threshold "
            "with strong conversion history. Order conservatively to avoid overstock."
        )
    else:
        recommendation = "Decline all signals — scores below the 65-point threshold. Monitor for improvement."

    confidence = "HIGH" if all(s["score"] > 70 or s["score"] < 55 for s in signals) else "MEDIUM"
    directive = json.dumps({"engine": "TREND", "actions": actions}, separators=(", ", ": "))

    return (
        f"SITUATION: {situation}\n\n"
        f"SIGNAL ANALYSIS: {signal_analysis}\n\n"
        f"VIABILITY CHECK: {' | '.join(viability_lines)}\n\n"
        f"RECOMMENDATION: {recommendation}\n\n"
        f"DIRECTIVE:\n{directive}\n\n"
        f"CONFIDENCE: {confidence}"
    )


def _make_signal(category: str, score: float, rng: random.Random, detected_at: int = 0) -> dict:
    # Distribute score into factors (rough normalisation)
    r = score / 100.0
    return {
        "category": category,
        "score": score,
        "source": rng.choice(_SIGNAL_SOURCES),
        "suggested_kg": rng.uniform(40, 150),
        "recipe_simplicity": min(r * rng.uniform(0.8, 1.2), 1.0),
        "ingredient_rarity": min(r * rng.uniform(0.6, 1.0), 1.0),
        "view_velocity": min(r * rng.uniform(0.7, 1.1), 1.0),
        "local_relevance": min(r * rng.uniform(0.5, 1.0), 1.0),
        "historical_conversion": min(r * rng.uniform(0.6, 1.0), 1.0),
        "detected_at": detected_at,
    }


def generate_trend_examples(n_per_difficulty: int = 10) -> list[dict]:
    examples = []
    rng = random.Random(3003)

    # EASY: 1 strong signal, clearly above threshold
    for i in range(n_per_difficulty):
        cat = rng.choice(_CATEGORIES)
        signals = [_make_signal(cat, rng.uniform(80, 95), rng, detected_at=rng.randint(0, 48))]
        inv = [_make_batch(f"TI{i}A", "FRESH", category=cat, rng=rng)]
        tick = rng.randint(50, 300)
        prompt = _trend_prompt(signals, inv, tick)
        examples.append({"engine_type": "TREND", "difficulty": "easy",
                         "prompt": prompt, "completion": _trend_completion(signals)})

    # MEDIUM: 2 signals, one above threshold
    for i in range(n_per_difficulty):
        cat_a = rng.choice(_CATEGORIES)
        cat_b = rng.choice([c for c in _CATEGORIES if c != cat_a])
        signals = [
            _make_signal(cat_a, rng.uniform(68, 82), rng, detected_at=rng.randint(0, 24)),
            _make_signal(cat_b, rng.uniform(50, 64), rng, detected_at=rng.randint(0, 48)),
        ]
        inv = [_make_batch(f"TMI{i}A", "WATCH", category=cat_a, rng=rng)]
        tick = rng.randint(50, 300)
        prompt = _trend_prompt(signals, inv, tick)
        examples.append({"engine_type": "TREND", "difficulty": "medium",
                         "prompt": prompt, "completion": _trend_completion(signals)})

    # HARD: 2-3 signals, borderline scores, low inventory
    for i in range(n_per_difficulty):
        cats = rng.sample(_CATEGORIES, 2)
        signals = [
            _make_signal(cats[0], rng.uniform(65, 75), rng, detected_at=rng.randint(0, 12)),
            _make_signal(cats[1], rng.uniform(55, 68), rng, detected_at=rng.randint(0, 24)),
        ]
        inv = []  # empty stock — no existing batches
        tick = rng.randint(100, 400)
        prompt = _trend_prompt(signals, inv, tick)
        examples.append({"engine_type": "TREND", "difficulty": "hard",
                         "prompt": prompt, "completion": _trend_completion(signals)})

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all(output_dir: str = "training/sft_data", n_per_difficulty: int = 10) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {
        "pricing_examples.json": generate_pricing_examples(n_per_difficulty),
        "farmer_examples.json": generate_farmer_examples(n_per_difficulty),
        "trend_examples.json": generate_trend_examples(n_per_difficulty),
    }

    total = 0
    for filename, examples in datasets.items():
        path = out / filename
        with open(path, "w") as f:
            json.dump(examples, f, indent=2)
        counts: dict[str, int] = {}
        for ex in examples:
            d = ex["difficulty"]
            counts[d] = counts.get(d, 0) + 1
        print(f"  {filename:35s} {len(examples):3d} examples | {counts}")
        total += len(examples)

    # Validate every completion contains all 6 required sections
    required = ["SITUATION:", "SIGNAL ANALYSIS:", "VIABILITY CHECK:",
                "RECOMMENDATION:", "DIRECTIVE:", "CONFIDENCE:"]
    bad = 0
    for filename, examples in datasets.items():
        for ex in examples:
            missing = [s for s in required if s not in ex["completion"]]
            if missing:
                print(f"  WARNING: {filename} example missing {missing}")
                bad += 1

    if bad:
        print(f"\nWARNING: {bad} examples failed section validation — check generate_sft_data.py")
    else:
        print(f"\nAll {total} examples pass section validation.")

    print(f"SFT data written to: {out.resolve()}")


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--output-dir", default="training/sft_data")
    parser.add_argument("--n-per-difficulty", type=int, default=10,
                        help="Examples per difficulty level (default 10, total = n*3*3 = 90)")
    args = parser.parse_args()

    print(f"Generating SFT data ({args.n_per_difficulty} per difficulty level)...")
    generate_all(output_dir=args.output_dir, n_per_difficulty=args.n_per_difficulty)
