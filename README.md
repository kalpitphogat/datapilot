# 📊 DataPilot — ask your data anything

**An AI data analyst powered by NVIDIA Nemotron on Nebius Token Factory, with Tavily web search.**
Upload a CSV or Excel file and ask questions in plain English. DataPilot writes and runs real Python
to find the answer, draws charts, and pulls in web context when your data alone isn't enough.

> Built for the Nebius x NVIDIA Global AI Hackathon 2026 — track: **Best Apps and Agents**.

## The problem
Most people who own data — small business owners, NGO staff, students, teachers — can't write pandas.
AI chatbots that "read" a spreadsheet often **hallucinate numbers**, which is worse than no answer.

## What DataPilot does
- **Answers grounded in executed code.** Every number in an answer comes from Python that actually ran.
  Expand **Show work** to see each step's code and output.
- **✨ One-click insight report.** The agent explores the whole dataset by itself — trends, seasonality,
  segments, relationships, data quality — and returns the top findings with charts and recommendations.
- **Web context via Tavily.** The model can call `web_search()` from inside its code, e.g. to compare your
  metrics with industry benchmarks, and cites its sources.
- **Follow-up suggestions** after each answer, and a conversation memory for drill-downs.
- **Shareable HTML report** with charts embedded — open it in a browser or print to PDF.

## How it works
```
question ──► NVIDIA Nemotron (Nebius Token Factory)
                │  writes one Python step
                ▼
          sandbox: df, pandas, numpy, scipy, matplotlib, web_search() ──► Tavily
                │  printed output / errors / charts  + remaining step budget
                ▼
          Nemotron reads the result → next step … → FINAL ANSWER
                │
                ▼
          answer reviewer: rejects claims about charts that weren't drawn,
          missing requested charts, and invented currencies → model revises
```

### Engineering details
| Piece | What it does |
|---|---|
| `agent.py` | Reason → code → observe loop. Errors are fed back so the model self-corrects. A reply containing code is always executed before any answer is accepted, so the model can't skip the evidence. A **step budget** is shown after each step, and an **answer reviewer** checks the final answer against what actually happened before accepting it. |
| `sandbox.py` | Persistent namespace like a notebook kernel. **60s timeout** per step. A **guard rail** blocks system imports, file reads and credential access, so a prompt-injected dataset can't leak API keys. Figures are captured and auto-styled with readable axis labels (`4.2M`, not `1e6`). |
| `search.py` | Tavily search exposed to the model as a Python function; sources are collected and cited. |
| `report.py` | Self-contained HTML report export. |
| `app.py` | Streamlit UI: live step-by-step progress, charts, sources, follow-ups. |
| `tests/` | Offline tests with a scripted fake model: error recovery, evidence-before-answer, chart-claim review, sandbox blocking, report export. |

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # add NEBIUS_API_KEY (and optionally TAVILY_API_KEY)
streamlit run app.py
```
Click **Try the sample sales dataset** to explore a synthetic Indian e-commerce dataset with hidden patterns:
a festive-season spike, a steadily declining region, weak marketing ROI, and a low-rated category.
Then hit **✨ Generate an insight report** and see how many it finds.

Run the tests with `pip install pytest && pytest`.

## Deploy (Streamlit Community Cloud)
Push to GitHub → share.streamlit.io → New app → `app.py`. Add `NEBIUS_API_KEY` and `TAVILY_API_KEY`
under **Secrets**.

⚠️ The sandbox guard rail blocks the obvious ways to touch the host, but it is not a hardened security
boundary. Use API keys with spending limits on a public deployment.

## Tech
NVIDIA Nemotron · Nebius Token Factory (OpenAI-compatible API) · Tavily · Streamlit · pandas · scipy · matplotlib

## License
MIT
