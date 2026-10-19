"""The DataPilot agent: NVIDIA Nemotron on Nebius Token Factory, reasoning in a code-execution loop."""
import re
from dataclasses import dataclass, field

import pandas as pd
from openai import OpenAI

from sandbox import Sandbox

# Both providers serve NVIDIA Nemotron through an OpenAI-compatible API.
# Nebius Token Factory is the hackathon's required platform; NVIDIA's API is handy for development.
PROVIDERS = {
    "Nebius Token Factory": "https://api.tokenfactory.nebius.com/v1/",
    "NVIDIA API (build.nvidia.com)": "https://integrate.api.nvidia.com/v1",
}

SYSTEM_PROMPT = """You are DataPilot, an expert data analyst. A user has uploaded a dataset and asks questions about it.
The dataset is already loaded in a pandas DataFrame named `df`.

Dataset profile:
{profile}

You work in steps, like a data scientist in a notebook. In each reply do exactly ONE of the following:
1. Write ONE ```python code block to run. Its printed output (or error) is sent back to you. Variables persist between steps.
2. When you have enough evidence, reply with a line starting with `FINAL ANSWER:` followed by your answer in Markdown.

Available in Python: df, pd, np, plt (matplotlib), scipy.stats (import it if needed), and
web_search(query, max_results=5) -> list of dicts with "title", "url", "content".
Use web_search only when outside context helps (industry benchmarks, definitions, explanations of real-world events).

Rules:
- print() everything you need to see. Only printed output comes back to you.
- Print summaries, not whole tables (aggregate, use .head(), round numbers).
- For a chart, create a matplotlib figure with a clear title and axis labels. Do not call plt.show() or savefig;
  figures are captured automatically and shown to the user. Make at most 2 charts per question.
  To compare groups, plot each group separately with label=... so colors and legend match.
- Never invent numbers. Every number in your final answer must come from code output you have seen.
- For "why" questions, dig for root causes: check how the metric changes over time (if there is a date column)
  and break it down by other dimensions before concluding.
- With time series, look for seasonal peaks and dips (which months stand out), not just a linear trend.
- Don't assume a currency or unit unless the data or user states it; report plain numbers with no unit label.
- If code fails, read the error and fix it. Note pandas 2.x: for monthly data use resample("MS")
  (month start, so points line up with their month on the x-axis), not "M".
- Watch your step budget (shown after each step). Save at least one step for any charts you need.
- Do not import os/sys/subprocess, read or write files, or access environment variables - such code is blocked.

Final answer format: start with the direct answer in 1-2 sentences, then key findings as bullet points with numbers,
then 1-2 concrete recommendations. Only mention a chart if an observation confirmed "chart(s) captured";
if the user asked for a chart and none was captured yet, make it before answering. If you used web_search, cite sources as [title](url).
End with a line `FOLLOW-UPS:` followed by 3 short follow-up questions the user could ask next, one per line starting with "- "."""

INSIGHT_REPORT_QUESTION = (
    "Create an executive insight report for this dataset. Plan: (1) in one step, compute an overview, data quality, "
    "and how the key metric moves over time (monthly, if there is a date) to spot trends and seasonality; "
    "(2) in 2-3 steps, break it down by segments and check relationships between variables, including whether "
    "any segment's trend differs from the rest; (3) in one step, draw the 2 charts that best show the top findings. "
    "Then present the 3-5 most important, non-obvious findings with numbers and end with concrete recommendations."
)

PREFERRED_MODEL = "nemotron-3-super-120b"  # shown first in the model picker when available
FORCE_FINAL = "You have used all your steps. Reply now with FINAL ANSWER: based only on the evidence so far."
CHART_WORDS = ("chart", "plot", "graph", "visuali")


@dataclass
class Step:
    code: str
    output: str
    error: str | None
    figures: list[bytes] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    sources: list[dict]
    follow_ups: list[str] = field(default_factory=list)

    @property
    def figures(self) -> list[bytes]:
        return [fig for step in self.steps for fig in step.figures]


def make_client(api_key: str, provider: str = "Nebius Token Factory") -> OpenAI:
    return OpenAI(base_url=PROVIDERS[provider], api_key=api_key)


def list_models(client: OpenAI) -> list[str]:
    """Chat model IDs available from the provider, with NVIDIA Nemotron models first."""
    non_chat = ("embed", "safety", "guard", "reward", "parse")
    ids = sorted(m.id for m in client.models.list().data if not any(k in m.id.lower() for k in non_chat))
    nemotron = sorted(
        (i for i in ids if "nemotron" in i.lower()),
        key=lambda i: (PREFERRED_MODEL not in i, i),
    )
    return nemotron + [i for i in ids if i not in nemotron]


def profile_dataframe(df: pd.DataFrame, max_chars: int = 6000) -> str:
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns", "Columns:"]
    for col in df.columns:
        series = df[col]
        examples = ", ".join(map(str, series.dropna().unique()[:5]))
        lines.append(f"- {col} ({series.dtype}, {int(series.isna().sum())} missing): e.g. {examples}")
    lines.append("\nFirst rows:\n" + df.head(5).to_string())
    return "\n".join(lines)[:max_chars]


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_code(text: str) -> str | None:
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def parse_answer(text: str) -> tuple[str, list[str]]:
    """Splits a final reply into the answer body and its suggested follow-up questions."""
    answer = text.split("FINAL ANSWER:", 1)[-1].strip()
    body, _, tail = answer.partition("FOLLOW-UPS:")
    follow_ups = [line.strip().lstrip("-*0123456789. ").strip() for line in tail.splitlines()]
    return body.strip().rstrip("*#").strip(), [q for q in follow_ups if q][:3]


def format_observation(step: Step) -> str:
    parts = []
    if step.output.strip():
        parts.append(f"Output:\n{step.output}")
    if step.error:
        parts.append(f"Error:\n{step.error}")
    if step.figures:
        parts.append(f"[{len(step.figures)} chart(s) captured and shown to the user]")
    if not parts:
        parts.append("(no output - remember to print() results)")
    return "\n\n".join(parts)


def review_answer(answer: str, question: str, steps: list[Step]) -> str | None:
    """Checks a final answer against what actually happened; returns feedback if it must be revised."""
    has_chart = any(step.figures for step in steps)
    if not has_chart and any(w in question.lower() for w in CHART_WORDS):
        return "The user asked for a chart but no chart has been captured yet. Create it now with matplotlib."
    if not has_chart and any(w in answer.lower() for w in CHART_WORDS):
        return "Your answer mentions a chart, but no chart was captured. Either create it now or remove the mention."
    if "$" in answer and "$" not in question:
        return "Don't assume the currency is dollars - the data doesn't say so. Rewrite the FINAL ANSWER without '$'."
    return None


def run_agent(
    client: OpenAI,
    model: str,
    question: str,
    sandbox: Sandbox,
    profile: str,
    history: list[dict] | None = None,
    max_steps: int = 6,
    on_step=None,
) -> AgentResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(profile=profile)},
        *(history or []),
        {"role": "user", "content": question},
    ]
    steps: list[Step] = []
    sources_start = len(sandbox.sources)

    def complete() -> str:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=4096
        )
        return strip_reasoning(response.choices[0].message.content or "")

    def finish(text: str) -> AgentResult:
        answer, follow_ups = parse_answer(text)
        if "$" not in question:  # last-resort guard if the model still assumes dollars
            answer = re.sub(r"\$(?=\s?[\d.,]+)", "", answer)
        return AgentResult(answer, steps, sandbox.sources[sources_start:], follow_ups)

    def budget_note(steps_left: int) -> str:
        note = f"[Step budget: {steps_left} step(s) left]"
        wants_chart = any(w in question.lower() for w in CHART_WORDS)
        if steps_left <= 2 and wants_chart and not any(s.figures for s in steps):
            note += " The requested chart(s) are not made yet - create them in your next step."
        elif steps_left <= 1:
            note += " Wrap up: reply with FINAL ANSWER next."
        return note

    revisions_left = 2
    for turn in range(max_steps):
        text = complete()
        code = extract_code(text)
        if code is None:
            feedback = review_answer(parse_answer(text)[0], question, steps) if revisions_left else None
            if feedback is None:
                return finish(text)
            revisions_left -= 1
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": feedback})
            continue

        # A reply with code always gets executed, even if it also contains a premature
        # FINAL ANSWER - the model must see real output before answering.
        result = sandbox.run(code)
        step = Step(code, result.output, result.error, result.figures)
        steps.append(step)
        if on_step:
            on_step(step)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"{format_observation(step)}\n\n{budget_note(max_steps - turn - 1)}"})

    messages.append({"role": "user", "content": FORCE_FINAL})
    return finish(complete())
