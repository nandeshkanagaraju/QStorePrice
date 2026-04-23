"""HuggingFace Spaces entry point — Gradio demo for QStorePrice AI.

Allows judges to interact with the environment without local install.
"""

from __future__ import annotations

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.enums import CurriculumScenario

# ---------------------------------------------------------------------------
# Global session state (one session at a time for the demo)
# ---------------------------------------------------------------------------
env: FreshPriceEnv | None = None
current_obs: str = ""
episode_wrr_history: list[float] = []
step_count: int = 0


def reset_episode(scenario_name: str, seed: int) -> tuple:
    """Reset the environment and return the initial observation."""
    global env, current_obs, episode_wrr_history, step_count

    scenario = CurriculumScenario[scenario_name]
    env = FreshPriceEnv(scenario=scenario, seed=int(seed))
    current_obs, info = env.reset()
    episode_wrr_history = [0.0]
    step_count = 0

    engine_type = info.get("engine_type", "PRICING")

    return (
        current_obs,
        _template_brief(engine_type),
        f"Episode reset. Scenario: {scenario_name} | Seed: {int(seed)} | First engine: {engine_type}",
        _wrr_chart([0.0]),
    )


def submit_brief(brief_text: str) -> tuple:
    """Submit a brief and advance the environment one step."""
    global current_obs, episode_wrr_history, step_count

    if env is None:
        return (
            "Reset the episode first.",
            "",
            "Not started. Press Reset Episode.",
            _wrr_chart([0.0]),
        )

    obs, reward, done, truncated, info = env.step(brief_text)
    step_count += 1

    # Track WRR from the state accumulator
    wrr = env._state.wrr if env._state is not None else 0.0
    episode_wrr_history.append(wrr)

    parse_ok = info.get("parse_success", True)
    quality = info.get("quality_score", 0.0)

    if done:
        final = info.get("final_reward", {})
        status = (
            f"Episode complete! "
            f"Final WRR: {final.get('wrr', 0):.4f} | "
            f"Brief Quality: {final.get('brief_quality_score', 0):.3f} | "
            f"Violations: {final.get('anti_hack_violations', 0)} | "
            f"Steps: {step_count}"
        )
        current_obs = "Episode complete. Press Reset to start a new episode."
        next_brief = ""
    else:
        parse_status = "OK" if parse_ok else "FAILED (using fallback)"
        status = (
            f"Step {step_count} | Reward: {reward:+.4f} | WRR: {wrr:.4f} | "
            f"Parse: {parse_status} | Quality: {quality:.3f}"
        )
        current_obs = obs
        engine_type = info.get("next_engine_type", "PRICING")
        next_brief = _template_brief(engine_type)

    return (
        current_obs,
        next_brief,
        status,
        _wrr_chart(episode_wrr_history),
    )


def _template_brief(engine_type: str) -> str:
    """Return a template brief for the given engine type."""
    templates = {
        "PRICING": (
            "SITUATION: [Describe current inventory state and urgency]\n"
            "SIGNAL ANALYSIS: N/A\n"
            "VIABILITY CHECK: N/A\n"
            "RECOMMENDATION: [Your pricing decision and justification]\n"
            'DIRECTIVE: {"engine": "PRICING", "actions": ['
            '{"batch_id": "BATCH_ID_HERE", "price_multiplier": 0.80, '
            '"flash_sale": false, "bundle_with": null}]}\n'
            "CONFIDENCE: MEDIUM"
        ),
        "FARMER": (
            "SITUATION: [Describe the farmer offer and current inventory]\n"
            "SIGNAL ANALYSIS: N/A\n"
            "VIABILITY CHECK: Shelf Life: PASS - adequate time. "
            "Conflict: PASS - no overlap. Break-Even: PASS - healthy margin. "
            "Worst-Case P&L: PASS - break-even at 60%. "
            "Demand Timing: PASS - good demand.\n"
            "RECOMMENDATION: [Accept/Counter/Decline with reason]\n"
            'DIRECTIVE: {"engine": "FARMER", "actions": ['
            '{"offer_id": "OFFER_ID_HERE", "decision": "ACCEPT", '
            '"counter_price": null}]}\n'
            "CONFIDENCE: HIGH"
        ),
        "TREND": (
            "SITUATION: [Describe the trend signal and current stock]\n"
            "SIGNAL ANALYSIS: [Describe the trend signal strength and likely demand impact]\n"
            "VIABILITY CHECK: Recipe Simplicity: PASS. Ingredient Rarity: PASS. "
            "View Velocity: PASS. Local Relevance: PASS. Historical Conversion: PASS.\n"
            "RECOMMENDATION: [Approve/Decline with reason]\n"
            'DIRECTIVE: {"engine": "TREND", "actions": ['
            '{"category": "CATEGORY_HERE", "decision": "APPROVE", '
            '"order_quantity_kg": 10.0}]}\n'
            "CONFIDENCE: HIGH"
        ),
    }
    return templates.get(engine_type, templates["PRICING"])


def _wrr_chart(history: list[float]) -> object:
    """Build a line chart of WRR over steps."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history, color="#3b82f6", linewidth=2)
    ax.axhline(
        y=0.70, color="#ef4444", linestyle="--",
        linewidth=1, label="Promotion threshold (0.70)",
    )
    ax.set_xlabel("Brief Cycle (Step)")
    ax.set_ylabel("WRR")
    ax.set_title("Weekly Waste Recovery Rate")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="QStorePrice AI — Perishable Goods Intelligence") as demo:
    gr.Markdown("# QStorePrice AI")
    gr.Markdown(
        "**Can an LLM learn to run a perishable goods store?** "
        "Write Operating Briefs to manage pricing, farmer offers, and "
        "trend-based restocking. "
        "Target: Weekly Waste Recovery Rate (WRR) >= 0.70."
    )

    with gr.Row():
        scenario_dropdown = gr.Dropdown(
            choices=[s.name for s in CurriculumScenario],
            value="STABLE_WEEK",
            label="Training Scenario",
        )
        seed_input = gr.Number(value=42, label="Episode Seed", precision=0)
        reset_btn = gr.Button("Reset Episode", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            obs_box = gr.Textbox(
                label="Current State (What the LLM sees)",
                lines=20,
                interactive=False,
            )
            brief_box = gr.Textbox(
                label="Your Operating Brief (Edit and submit)",
                lines=15,
                interactive=True,
            )
            submit_btn = gr.Button("Submit Brief", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=1):
            wrr_plot = gr.Plot(label="WRR Progress")

    reset_btn.click(
        fn=reset_episode,
        inputs=[scenario_dropdown, seed_input],
        outputs=[obs_box, brief_box, status_box, wrr_plot],
    )
    submit_btn.click(
        fn=submit_brief,
        inputs=[brief_box],
        outputs=[obs_box, brief_box, status_box, wrr_plot],
    )

    gr.Markdown("""
## How to Use
1. Select a scenario and press **Reset Episode**
2. Read the state observation carefully
3. Edit the pre-filled Operating Brief template
4. Replace placeholder IDs (BATCH_ID_HERE, OFFER_ID_HERE) with real IDs from the state
5. Press **Submit Brief** to advance the simulation
6. Watch the WRR chart — target is above the red dashed line (0.70)

## The Six Brief Sections
- **SITUATION**: What is happening right now (inventory, urgency, signals)
- **SIGNAL ANALYSIS**: Trend signal details (TREND briefs only, else N/A)
- **VIABILITY CHECK**: 5-factor check (FARMER/TREND briefs only, else N/A)
- **RECOMMENDATION**: Your decision with one-sentence justification
- **DIRECTIVE**: Machine-readable JSON — the rule executor acts on this
- **CONFIDENCE**: HIGH / MEDIUM / LOW
""")


if __name__ == "__main__":
    demo.launch()
