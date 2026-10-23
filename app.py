"""DataPilot - ask your data anything. Streamlit UI."""
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import INSIGHT_REPORT_QUESTION, PROVIDERS, make_client, list_models, profile_dataframe, run_agent
from report import build_html_report
from sandbox import Sandbox
from search import make_web_search

load_dotenv()
st.set_page_config(page_title="DataPilot", page_icon="📊", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sales.csv"
SUGGESTIONS = [
    "Which region is underperforming, and why?",
    "Is marketing spend actually driving revenue? Show a chart.",
    "How do sales change across the year? Any seasonality?",
    "Which product category has the lowest customer rating, and how does that compare to industry benchmarks?",
]
HISTORY_TURNS = 4
# The public demo runs on the owner's API keys, so cap usage per browser session.
MAX_QUESTIONS_PER_SESSION = 15
INSIGHT_REPORT_LABEL = "✨ Generate an insight report for this dataset"

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1100px; }
      .dp-hero h1 { font-size: 2.6rem; margin-bottom: 0; }
      .dp-hero p { font-size: 1.1rem; color: #6b7280; }
      .dp-badges span { display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0; border-radius: 999px;
                        font-size: 0.8rem; border: 1px solid rgba(118,185,0,.5); color: #4d7c0f; background: rgba(118,185,0,.08); }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret(name: str) -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, "")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_models(api_key: str, provider: str) -> list[str]:
    return list_models(make_client(api_key, provider))


def load_file(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)
    for col in df.columns:
        if df[col].dtype == object and any(k in col.lower() for k in ("date", "time", "month")):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[col] = parsed
    return df


def set_dataset(name: str, df: pd.DataFrame) -> None:
    if st.session_state.get("dataset_name") == name:
        return
    st.session_state.dataset_name = name
    st.session_state.df = df
    st.session_state.profile = profile_dataframe(df)
    st.session_state.sandbox = Sandbox(df.copy(), make_web_search(tavily_key))
    st.session_state.messages = []


def ask(question: str) -> None:
    st.session_state.pending_question = question


def render_step(i: int, step) -> None:
    st.markdown(f"**Step {i}**")
    st.code(step.code, language="python")
    if step.output.strip():
        st.code(step.output, language="text")
    if step.error:
        st.error(step.error.strip().splitlines()[-1])


def render_reply(msg: dict, key: str, show_follow_ups: bool) -> None:
    st.markdown(msg["content"])
    figures = msg.get("figures", [])
    if figures:
        cols = st.columns(min(len(figures), 2))
        for i, fig in enumerate(figures):
            cols[i % len(cols)].image(fig)
    if msg.get("sources"):
        with st.expander(f"🌐 Web sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                st.markdown(f"- [{s.get('title') or s['url']}]({s['url']})")
    if msg.get("steps"):
        with st.expander(f"🔍 Show work · {len(msg['steps'])} steps · {msg.get('seconds', 0):.0f}s"):
            for i, step in enumerate(msg["steps"], 1):
                render_step(i, step)
    if show_follow_ups and msg.get("follow_ups"):
        st.caption("Ask next")
        for i, q in enumerate(msg["follow_ups"]):
            st.button(q, key=f"{key}-f{i}", on_click=ask, args=(q,))


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📊 DataPilot")
    st.caption("Your AI data analyst")

    provider = st.selectbox("Provider", list(PROVIDERS), index=0 if secret("NEBIUS_API_KEY") or not secret("NVIDIA_API_KEY") else 1)
    key_name = "NEBIUS_API_KEY" if provider.startswith("Nebius") else "NVIDIA_API_KEY"
    api_key = secret(key_name) or st.text_input(f"{provider} API key", type="password")
    tavily_key = secret("TAVILY_API_KEY") or st.text_input("Tavily API key (optional, for web context)", type="password")

    model = None
    if api_key:
        try:
            models = cached_models(api_key, provider)
            default = secret("DEFAULT_MODEL")
            model = st.selectbox("Model", models, index=models.index(default) if default in models else 0)
        except Exception as exc:
            st.error(f"Could not reach {provider}: {exc}")
    st.caption(("🟢 Web context on" if tavily_key else "⚪ Web context off") + " · Tavily")

    st.divider()
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            set_dataset(uploaded.name, load_file(uploaded))
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
    elif st.button("Try the sample sales dataset", width="stretch"):
        set_dataset("sample: sales.csv", pd.read_csv(SAMPLE_PATH, parse_dates=["date"]))

    if st.session_state.get("messages"):
        st.divider()
        st.download_button(
            "⬇️ Download report (HTML)",
            build_html_report(st.session_state.dataset_name, st.session_state.messages),
            file_name="datapilot_report.html",
            mime="text/html",
            width="stretch",
        )
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.rerun()

# ---------- Main ----------
if "df" not in st.session_state:
    st.markdown(
        """
        <div class="dp-hero">
          <h1>Ask your data anything.</h1>
          <p>An AI data analyst that writes and runs real Python, draws charts, and checks the web for context.
          Every number comes from executed code, not guesses.</p>
          <div class="dp-badges"><span>NVIDIA Nemotron</span><span>Nebius Token Factory</span><span>Tavily web search</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Upload**\n\nA CSV or Excel file, or try the sample dataset from the sidebar.")
    c2.info("**2. Ask**\n\n\"Why did sales drop in Q3?\" or one click for a full insight report.")
    c3.info("**3. Get answers**\n\nInsights, charts, sources, and the code behind every number.")
    st.stop()

df = st.session_state.df
with st.expander(f"📄 {st.session_state.dataset_name} · {df.shape[0]:,} rows × {df.shape[1]} columns",
                 expanded=not st.session_state.messages):
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    m = st.columns(4)
    m[0].metric("Rows", f"{df.shape[0]:,}")
    m[1].metric("Columns", df.shape[1])
    m[2].metric("Missing cells", f"{df.isna().mean().mean():.1%}")
    m[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    if date_cols:
        span = df[date_cols[0]]
        st.caption(f"🗓️ `{date_cols[0]}` spans {span.min():%d %b %Y} → {span.max():%d %b %Y}")
    st.dataframe(df.head(50), width="stretch")

messages = st.session_state.messages
for idx, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_reply(msg, key=f"m{idx}", show_follow_ups=idx == len(messages) - 1)

if not messages:
    st.button(INSIGHT_REPORT_LABEL, type="primary", width="stretch", on_click=ask, args=(INSIGHT_REPORT_LABEL,))
    st.caption("…or ask your own question:")
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        cols[i % 2].button(suggestion, key=f"s{i}", width="stretch", on_click=ask, args=(suggestion,))

question = st.chat_input("Ask a question about your data…") or st.session_state.pop("pending_question", None)

if question:
    if not model:
        st.warning("Add an API key in the sidebar first.")
        st.stop()
    asked = st.session_state.get("questions_asked", 0)
    if asked >= MAX_QUESTIONS_PER_SESSION:
        st.warning(f"This public demo allows {MAX_QUESTIONS_PER_SESSION} questions per session. "
                   "Run it locally from the GitHub repo for unlimited use.")
        st.stop()
    st.session_state.questions_asked = asked + 1

    is_report = question == INSIGHT_REPORT_LABEL
    prompt = INSIGHT_REPORT_QUESTION if is_report else question
    history = [{"role": m["role"], "content": m["content"]} for m in messages[-2 * HISTORY_TURNS:]]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.time()
        with st.status("Analyzing…", expanded=True) as status:
            step_count = [0]

            def on_step(step):
                step_count[0] += 1
                status.update(label=f"Analyzing… step {step_count[0]}")
                render_step(step_count[0], step)

            try:
                result = run_agent(
                    make_client(api_key, provider), model, prompt,
                    st.session_state.sandbox, st.session_state.profile,
                    history=history, max_steps=8 if is_report else 6, on_step=on_step,
                    fallback_models=[m for m in models if "nemotron" in m.lower()],
                )
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.error(f"Model call failed: {exc}")
                messages.pop()
                st.stop()
            status.update(label=f"Done · {len(result.steps)} steps · {time.time() - started:.0f}s",
                          state="complete", expanded=False)

        reply = {
            "role": "assistant",
            "content": result.answer,
            "steps": result.steps,
            "figures": result.figures,
            "sources": result.sources,
            "follow_ups": result.follow_ups,
            "seconds": time.time() - started,
        }
        render_reply(reply, key=f"m{len(messages)}", show_follow_ups=True)
    messages.append(reply)
