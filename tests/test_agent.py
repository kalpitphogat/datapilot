"""Offline tests for the agent loop, sandbox, and report - a scripted fake model stands in for Nemotron."""
from types import SimpleNamespace

import pandas as pd
import pytest

from agent import parse_answer, review_answer, run_agent, strip_reasoning, Step
from report import build_html_report
from sandbox import Sandbox, find_blocked


class FakeModel:
    """Replays scripted replies and records the messages it was sent."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(list(kwargs["messages"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.replies)))])


@pytest.fixture
def df():
    return pd.DataFrame({"region": ["East", "West", "East", "North"], "revenue": [10, 40, 20, 30]})


def no_search(query, max_results=5):
    return []


def run(df, replies, question="Which region is weakest?"):
    model = FakeModel(replies)
    result = run_agent(model, "fake", question, Sandbox(df, no_search), "profile")
    return model, result


def test_recovers_from_error_and_answers(df):
    model, result = run(df, [
        "```python\nprint(df.groupby('regionX').revenue.sum())\n```",
        "```python\nprint(df.groupby('region').revenue.sum())\n```",
        "FINAL ANSWER: East is weakest.\nFOLLOW-UPS:\n- Why?\n- Show a trend\n- Compare to West",
    ])
    assert [bool(s.error) for s in result.steps] == [True, False]
    assert "KeyError" in model.calls[1][-1]["content"]
    assert result.answer == "East is weakest."
    assert result.follow_ups == ["Why?", "Show a trend", "Compare to West"]


def test_code_is_executed_before_premature_answer(df):
    _, result = run(df, [
        "```python\nprint(df.revenue.sum())\n```\nFINAL ANSWER: total is 999",
        "FINAL ANSWER: total is 100",
    ])
    assert result.steps[0].output.strip() == "100"
    assert result.answer == "total is 100"


def test_rejects_answer_claiming_missing_chart(df):
    model, result = run(df, [
        "FINAL ANSWER: See the chart above.",
        "```python\ndf.groupby('region').revenue.sum().plot(kind='bar', title='Revenue')\n```",
        "FINAL ANSWER: East is weakest (see chart).",
    ], question="Which region is weakest?")
    assert "no chart was captured" in model.calls[1][-1]["content"]
    assert len(result.figures) == 1


def test_variables_persist_between_steps(df):
    sandbox = Sandbox(df, no_search)
    sandbox.run("total = df.revenue.sum()")
    assert sandbox.run("print(total * 2)").output.strip() == "200"


@pytest.mark.parametrize("code", [
    "import os\nprint(os.environ)",
    "from subprocess import run",
    "print(open('.env').read())",
    "pd.read_csv('.env')",
    "__import__('os')",
    "eval('1+1')",
])
def test_sandbox_blocks_unsafe_code(df, code):
    assert find_blocked(code)
    assert "SecurityError" in Sandbox(df, no_search).run(code).error


def test_sandbox_allows_normal_analysis(df):
    code = "import re\nx = re.compile('E').match('East')\nprint(df.describe())"
    assert find_blocked(code) is None


def test_web_search_sources_are_collected(df):
    results = [{"title": "Benchmark", "url": "https://example.com", "content": "4.2 stars"}]
    sandbox = Sandbox(df, lambda q, n=5: results)
    sandbox.run("web_search('ratings benchmark')")
    assert sandbox.sources == results


def test_review_and_parsing_helpers():
    assert strip_reasoning("<think>hmm</think>Hello") == "Hello"
    assert review_answer("Revenue is $5", "how much?", []) is not None
    assert review_answer("Revenue is 5", "how much?", [Step("", "", None)]) is None
    assert parse_answer("FINAL ANSWER: Done.") == ("Done.", [])


def test_html_report_embeds_charts():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "**Bold** answer", "figures": [b"png-bytes"],
         "sources": [{"title": "Src", "url": "https://example.com"}]},
    ]
    html = build_html_report("data.csv", messages)
    assert "<strong>Bold</strong>" in html and "data:image/png;base64," in html and "https://example.com" in html
