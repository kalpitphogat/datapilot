## Inspiration
Almost everyone has a spreadsheet they can't fully use: shop owners with sales sheets, NGO staff with survey results, students with lab data. They know the questions — *"Why did sales drop?"*, *"Which region is struggling?"* — but not pandas.

AI chatbots look like the answer, but when you ask them about a spreadsheet they often **make numbers up**. For data analysis, a confident wrong number is worse than no answer. I wanted an analyst that works the way a real data scientist does: write code, run it, look at the result, and only then answer, with the work shown.

## What it does
**DataPilot** is an AI data analyst. Upload a CSV or Excel file and ask questions in plain English.

- 🧠 **Answers grounded in executed code.** An NVIDIA Nemotron agent writes Python, runs it on your data, reads the output, and repeats until it has evidence. Every number in the final answer comes from code that actually ran. Click **Show work** to see each step's code and output.
- ✨ **One-click insight report.** The agent explores the whole dataset by itself: trends over time, seasonality, segments that behave differently, relationships between variables, and data quality. It returns the top findings with charts and recommendations.
- 🌐 **Web context with Tavily.** Some questions can't be answered from the data alone, like *"Is a 3.4 rating bad for electronics?"*. The agent can call `web_search()` from inside its own code to pull in industry benchmarks, and it cites its sources.
- 📊 **Charts, follow-ups and a shareable report.** Charts are drawn automatically, every answer suggests three follow-up questions, and the whole conversation exports as an HTML report with the charts embedded.

On the included sample dataset (Indian e-commerce sales), the insight report finds the **October–November festive peak (+80% vs. the Jan–Sep average)**, the **East region's steady decline**, and the **weak link between marketing spend and revenue (r ≈ 0.2)**, all without being told where to look.

## How I built it
```
question → Nemotron (Nebius Token Factory) → writes one Python step
        → sandbox runs it (df, pandas, scipy, matplotlib, web_search → Tavily)
        → output / errors / charts + remaining step budget go back to Nemotron
        → … repeat … → FINAL ANSWER → answer reviewer → user
```

- **Model:** NVIDIA Nemotron (`nemotron-3.5-super-120b` by default) served through **Nebius Token Factory's** OpenAI-compatible API. Hosted catalogs retire models without notice (one of mine disappeared mid-hackathon), so if a model returns "not found" DataPilot automatically moves on to the next available Nemotron model.
- **Agent loop (`agent.py`):** a reason → code → observe loop with up to 6 steps (8 for reports). Errors go back to the model so it fixes its own code. A reply that contains code is **always executed before any answer is accepted**, so the model can't skip the evidence.
- **Answer reviewer:** before an answer reaches the user, it is checked against what actually happened. A **number grounding check** traces every number in the answer back to code output or a web source, allowing for rounding and %/K/M scaling. If a number can't be traced, or the model claims a chart it never drew, skips a requested chart, or invents a currency, the answer is sent back for revision. If a number still can't be traced after several revisions, the answer shows it under a ⚠️ **Unverified** label instead of presenting it as fact.
- **Sandbox (`sandbox.py`):** a persistent namespace, like a notebook kernel, so follow-up questions build on earlier work. It has a 60-second timeout per step and a guard rail that blocks system imports, file reads and access to credentials, so a malicious dataset can't trick the model into leaking API keys.
- **Tavily (`search.py`):** exposed to the model as a plain Python function, so it can mix web results with computations in the same step.
- **UI:** Streamlit, with live step-by-step progress, charts, sources, follow-up buttons and HTML export.
- **Tests:** 29 offline tests that use a scripted fake model to check error recovery, evidence-before-answer, chart-claim review, sandbox blocking and report export.

## Challenges I ran into
Most of the work was making the agent **honest**, not just capable. Each of these came from real test runs:

1. **Numbers that came from nowhere.** While recording the demo I caught Nemotron reporting the right rating for the worst category but made-up ratings for the others. That's exactly the failure DataPilot exists to prevent. So I built a **number grounding check**: every number in an answer must trace back to code output or a web source. In the next live run it flagged 4 untraceable numbers, the model computed them properly, and every figure in the final answer was correct. Numbers that still can't be traced after revisions are labelled ⚠️ Unverified. It's a strong filter, not a proof: a made-up number that happens to equal some printed value can still slip through.
2. **Charts that didn't exist.** The model would write *"as the chart above shows"* when it never drew one. Asking nicely in the prompt didn't fix it reliably, so I added a code-level **answer reviewer** that compares the answer with what actually ran and sends it back if they don't match.
3. **Running out of steps.** On open-ended reports the model kept exploring and hit the step limit without drawing any charts. Showing a **step budget** after every step, with a nudge when charts are still missing, fixed this.
4. **Truncated tables.** pandas hid wide tables behind `...`, so the model couldn't see some columns and repeated the same code over and over. Widening the display options fixed it.
5. **Invented currencies.** It labelled rupee data with `$`. Now currencies are never assumed unless the data says so.
6. **Library drift.** The model used outdated pandas syntax (`resample("M")`). Feeding errors back plus a short tip in the prompt lets it recover on its own.
7. **Running model-written code safely.** I added a timeout and a guard rail, while still allowing normal analysis code.

## Accomplishments that I'm proud of
- On the sample dataset, the agent finds the planted patterns (seasonality, a declining region, weak marketing ROI) without being told where to look.
- Answers are **traceable**: every claim can be checked by expanding the code behind it.
- Readable, consistent charts with labels like `2.0M` instead of `1e6`.

## What I learned
- With LLM agents, **checking the output in code beats instructing in prompts**. Prompts reduce mistakes; code-level checks remove whole categories of them.
- Letting the model **see its own errors** is the cheapest reliability boost there is.
- Reasoning models like Nemotron are strong at multi-step analysis when the loop gives them clear feedback.

## What's next for DataPilot
- Connect to databases and Google Sheets, not just file uploads.
- Scheduled reports, e.g. "email me this analysis every Monday".
- Multi-file analysis that joins related spreadsheets.
- A stronger sandbox, using containers per session.
- Support for Indian languages, so more small business owners can ask questions in Hindi and other languages.
