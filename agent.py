"""The DataPilot agent: NVIDIA Nemotron on Nebius Token Factory, reasoning in a code-execution loop."""
import json
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
  Never type figures from memory into code (e.g. printing "benchmark: 4.2"); numbers you write into code
  yourself are not evidence. For outside facts, print what web_search actually returns.
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
FORMAT_REMINDER = ("Your reply had no answer in the required format. Reply with exactly ONE ```python code block, "
                   "or a line starting with FINAL ANSWER: followed by your answer.")
GAVE_UP = ("FINAL ANSWER: I couldn't finish this analysis reliably. "
           "Expand **Show work** to see what I found so far, or try a more specific question.")
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
    # Hosted inference latency can spike; cap each call and retry rather than hang the whole analysis.
    return OpenAI(base_url=PROVIDERS[provider], api_key=api_key, timeout=120, max_retries=2)


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
    if match:
        return match.group(1).strip()
    # Nemotron sometimes wraps a step as JSON instead, e.g. {"action": "python", "code": "..."}.
    start = text.find("{")
    if start != -1 and '"code"' in text:
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(text[start:])
        except ValueError:
            return None
        code = obj.get("code") if isinstance(obj, dict) else None
        if isinstance(code, str) and code.strip():
            return code.strip()
    return None


def parse_answer(text: str) -> tuple[str, list[str]]:
    """Splits a final reply into the answer body and its suggested follow-up questions."""
    # Models often bold the markers ("**FINAL ANSWER:**"), so swallow surrounding * and #.
    answer = re.split(r"[*#]*\s*FINAL ANSWER:\**", text, maxsplit=1)[-1]
    body, *tail = re.split(r"[*#\s]*FOLLOW-UPS\**:?\**", answer, maxsplit=1)
    follow_ups = [line.strip().lstrip("-*0123456789. ").strip() for line in "".join(tail).splitlines()]
    return body.strip(), [q for q in follow_ups if q][:3]


def looks_like_prose(text: str) -> bool:
    """True for a plausible plain-language answer; False for code, JSON, or degenerate repeated output."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(text) < 40 or '{"' in text or "{ \"" in text or "print(" in text:
        return False
    return len(set(lines)) >= 0.7 * len(lines)


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


NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(\s?(?:%|[KMB]\b|thousand|million|billion))?", re.IGNORECASE)
# How an evidence value may be rescaled to match, depending on the unit written after the answer's number.
SCALES = {"%": (1, 100), "k": (1, 1e-3), "thousand": (1, 1e-3), "m": (1, 1e-6), "million": (1, 1e-6),
          "b": (1, 1e-9), "billion": (1, 1e-9)}


def _numbers(text: str) -> list[tuple[float, int, str]]:
    """(value, decimal places, unit suffix) for every number in the text."""
    found = []
    for match in NUMBER.finditer(text):
        suffix = (match.group(1) or "").strip().lower()
        token = match.group()[: len(match.group()) - len(match.group(1) or "")].replace(",", "")
        try:
            found.append((float(token), len(token.split(".")[1]) if "." in token else 0, suffix))
        except ValueError:
            pass
    return found


def computed_output(step: "Step") -> str:
    """The numbers a step's output actually computed.

    A number that the same step's code typed in as a literal is dropped: a model can print a made-up
    "benchmark" from code, and that output must not count as a computed result.
    """
    typed = {value for value, _, _ in _numbers(step.code)}
    return " ".join(m.group() for m in NUMBER.finditer(step.output)
                    if _numbers(m.group()) and _numbers(m.group())[0][0] not in typed)


def ungrounded_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers in the answer that can't be traced to the evidence (allowing rounding, and %/K/M/B when written)."""
    answer = re.sub(r"\]\(https?://[^)]*\)", "]", answer)  # ignore numbers inside link URLs
    known = {value for value, _, _ in _numbers(evidence)}
    missing = []
    for value, decimals, suffix in _numbers(answer):
        if decimals == 0 and not suffix and (abs(value) <= 12 or 1900 <= value <= 2100):
            continue  # small counts ("3 findings"), months, and years
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        # Compare magnitudes: models often write negatives with a typographic dash ("–4.2%"), which isn't parsed as minus.
        if not any(abs(abs(value) - abs(k) * s) <= tolerance for k in known for s in SCALES.get(suffix, (1,))):
            missing.append(f"{value:g}{suffix if suffix == '%' else ''}")
    return list(dict.fromkeys(missing))


WEB_WORDS = ("benchmark", "industry", "market average", "competitor", "national average", "typical for")
CITATION = re.compile(r"【|\]\(https?://")


def review_answer(answer: str, question: str, steps: list[Step], evidence: str = "",
                  searched: bool = True) -> str | None:
    """Checks a final answer against what actually happened; returns feedback if it must be revised."""
    if not answer.strip():
        return "Your reply contained no answer. Reply now with FINAL ANSWER: followed by your findings."
    has_chart = any(step.figures for step in steps)
    if not has_chart and any(w in question.lower() for w in CHART_WORDS):
        return "The user asked for a chart but no chart has been captured yet. Create it now with matplotlib."
    if not has_chart and any(w in answer.lower() for w in CHART_WORDS):
        return "Your answer mentions a chart, but no chart was captured. Either create it now or remove the mention."
    if "$" in answer and "$" not in question:
        return "Don't assume the currency is dollars - the data doesn't say so. Rewrite the FINAL ANSWER without '$'."
    if not searched and any(w in question.lower() for w in WEB_WORDS):
        return ("The user asked for outside context (benchmarks/industry), but you haven't called web_search. "
                "Call web_search() in a code step and base the comparison on what it returns.")
    if not searched and CITATION.search(answer):
        return "Your answer cites sources, but you never called web_search. Search first, or remove the citations."
    missing = ungrounded_numbers(answer, evidence)
    if missing:
        return (f"These numbers in your answer don't appear in any code output or web source: {', '.join(missing[:10])}. "
                "Every number must come from output you have seen. Compute them with code now, or remove them.")
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
    # Earlier answers are stored without the marker; restore it so the model keeps following the format.
    past = [{**m, "content": "FINAL ANSWER: " + m["content"]} if m["role"] == "assistant" else m for m in history or []]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(profile=profile)},
        *past,
        {"role": "user", "content": question},
    ]
    steps: list[Step] = []
    sources_start = len(sandbox.sources)

    def complete() -> str:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=8192
        )
        return strip_reasoning(response.choices[0].message.content or "")

    def gather_evidence() -> str:
        # The profile is deliberately left out: its sample rows contain so many small values
        # that made-up figures (e.g. "a 4.3-star benchmark") would match it by coincidence.
        return "\n".join([question, *(computed_output(s) for s in steps),
                          *(src.get("content", "") for src in sandbox.sources[sources_start:])])

    def finish(text: str) -> AgentResult:
        answer, follow_ups = parse_answer(text)
        if "$" not in question:  # last-resort guard if the model still assumes dollars
            answer = re.sub(r"\$(?=\s?[\d.,]+)", "", answer)
        # If revisions ran out with numbers still untraceable, say so rather than pass them off as verified.
        unverified = ungrounded_numbers(answer, gather_evidence()) if text != GAVE_UP else []
        if unverified:
            answer += ("\n\n> ⚠️ **Unverified:** these numbers could not be traced to any code output or web source: "
                       + ", ".join(unverified[:10]))
        return AgentResult(answer, steps, sandbox.sources[sources_start:], follow_ups)

    def budget_note(steps_left: int) -> str:
        note = f"[Step budget: {steps_left} step(s) left]"
        wants_chart = any(w in question.lower() for w in CHART_WORDS)
        if steps_left <= 2 and wants_chart and not any(s.figures for s in steps):
            note += " The requested chart(s) are not made yet - create them in your next step."
        elif steps_left <= 1:
            note += " Wrap up: reply with FINAL ANSWER next."
        return note

    # Only executed code steps use the step budget; format and review revisions have their own limit.
    revisions_left = 3
    while len(steps) < max_steps:
        text = complete()
        code = extract_code(text)
        if code is None:
            # Without the marker the reply is malformed (e.g. a degenerate, repeating output), not an answer,
            # and is never shown to the user.
            if "FINAL ANSWER:" not in text:
                if not revisions_left:
                    return finish("FINAL ANSWER: " + text if looks_like_prose(text) else GAVE_UP)
                feedback = FORMAT_REMINDER
            else:
                searched = len(sandbox.sources) > sources_start
                feedback = (review_answer(parse_answer(text)[0], question, steps, gather_evidence(), searched)
                            if revisions_left else None)
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
        messages.append({"role": "user", "content": f"{format_observation(step)}\n\n{budget_note(max_steps - len(steps))}"})

    messages.append({"role": "user", "content": FORCE_FINAL})
    text = complete()
    return finish(text if "FINAL ANSWER:" in text else GAVE_UP)
