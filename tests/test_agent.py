"""Offline tests for the agent loop, sandbox, and report - a scripted fake model stands in for Nemotron."""
from types import SimpleNamespace

import pandas as pd
import pytest

from agent import parse_answer, review_answer, run_agent, strip_reasoning, ungrounded_numbers, Step
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
    sandbox.run("web_search('ratings benchmark again')")
    assert sandbox.sources == results  # the same URL is listed once


def test_rejects_numbers_not_in_any_output(df):
    model, result = run(df, [
        "```python\nprint(df.groupby('region').revenue.mean())\n```",
        "FINAL ANSWER: East averages 15, West averages 47.5.",
        "```python\nprint(df[df.region == 'West'].revenue.mean())\n```",
        "FINAL ANSWER: East averages 15, West averages 40.",
    ])
    assert "47.5" in model.calls[2][-1]["content"]
    assert result.answer == "East averages 15, West averages 40."


def test_benchmark_question_requires_a_web_search(df):
    model, _ = run(df, ["FINAL ANSWER: Below the industry average [Report](https://x.com)."] * 4,
                   question="How does East compare to industry benchmarks?")
    assert "web_search" in model.calls[1][-1]["content"]


def test_citations_without_search_are_rejected():
    assert "never called web_search" in review_answer("Avg is good【Jungle Scout】", "how is it?", [Step("", "", None)], searched=False)


def test_numbers_printed_from_code_literals_are_not_evidence(df):
    model, _ = run(df, [
        "```python\nprint('Statista benchmark: 4.6')\n```",
        "FINAL ANSWER: The benchmark is 4.6.",
        "FINAL ANSWER: I could not find a benchmark.",
    ])
    assert "4.6" in model.calls[2][-1]["content"]


def test_number_computed_earlier_stays_evidence_when_reused_as_literal_later(df):
    model, result = run(df, [
        "```python\nprint(round(df.revenue.mean(), 3))\n```",
        "```python\nmean = 25.0\nprint('gap', 30 - mean)\n```",
        "FINAL ANSWER: The mean is 25.0.",
    ])
    assert result.answer == "The mean is 25.0."


def test_falls_back_when_a_model_is_retired(df):
    import httpx
    from openai import NotFoundError

    class RetiredFirst(FakeModel):
        def _create(self, **kwargs):
            if kwargs["model"] == "retired":
                response = httpx.Response(404, request=httpx.Request("POST", "https://x"))
                raise NotFoundError("not found", response=response, body=None)
            return super()._create(**kwargs)

    model = RetiredFirst(["FINAL ANSWER: ok from backup"])
    result = run_agent(model, "retired", "Q", Sandbox(df, no_search), "profile", fallback_models=["retired", "backup"])
    assert result.answer == "ok from backup"


def test_revisions_do_not_use_up_the_step_budget(df):
    _, result = run(df, ["no marker"] * 3 + ["```python\nprint(df.revenue.sum())\n```", "FINAL ANSWER: Total is 100."])
    assert result.answer == "Total is 100." and len(result.steps) == 1


def test_numbers_still_unverified_after_revisions_are_flagged(df):
    _, result = run(df, ["FINAL ANSWER: The industry benchmark is 76."] * 4)
    assert result.answer.startswith("The industry benchmark is 76.")
    assert "Unverified" in result.answer and "76" in result.answer.split("Unverified")[1]


def test_grounding_allows_rounding_percent_and_scaling():
    evidence = "share 0.38214\nrevenue 2033305.4\nrating 3.4243"
    assert ungrounded_numbers("38.2% share, 2.03M revenue, 2,033,305 total, rating 3.42, top 3 in 2025", evidence) == []
    assert ungrounded_numbers("rating 3.80", evidence) == ["3.8"]
    # A fraction only matches a percentage when the answer actually writes "%".
    assert ungrounded_numbers("38 stores", "share 0.38") == ["38"]
    # Negatives written with a typographic dash still match.
    assert ungrounded_numbers("East –62.4%, corr –0.41", "East -0.6241\ncorr -0.406") == []


def test_review_and_parsing_helpers():
    assert strip_reasoning("<think>hmm</think>Hello") == "Hello"
    assert review_answer("Revenue is $5", "how much?", []) is not None
    assert review_answer("Revenue is 5", "how much?", [Step("", "", None)]) is None
    assert parse_answer("FINAL ANSWER: Done.") == ("Done.", [])
    assert parse_answer("**FINAL ANSWER:** **East** wins.\n\n**FOLLOW-UPS:**\n- Why?") == ("**East** wins.", ["Why?"])
    assert parse_answer("FINAL ANSWER: **Report**\nBody") == ("**Report**\nBody", [])
    assert parse_answer("FINAL ANSWER: Body\n\n**FOLLOW-UPS**\n- Next?") == ("Body", ["Next?"])


def test_malformed_reply_is_not_accepted_as_answer(df):
    junk = '{ "role": "assistant", "content": "python\\nprint(df)" }' * 3
    model, result = run(df, [junk, "FINAL ANSWER: East is weakest."])
    assert "required format" in model.calls[1][-1]["content"]
    assert result.answer == "East is weakest."


def test_malformed_replies_never_reach_the_user(df):
    _, result = run(df, ['{ "code": 1 } garbage'] * 4)
    assert "couldn't finish" in result.answer and "garbage" not in result.answer


def test_prose_without_marker_is_accepted_after_reminders(df):
    prose = "East is the weakest region overall, and it declined through the year."
    _, result = run(df, [prose] * 4)
    assert result.answer == prose


def test_history_answers_keep_the_marker(df):
    model = FakeModel(["FINAL ANSWER: ok"])
    history = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
    run_agent(model, "fake", "Q2", Sandbox(df, no_search), "profile", history=history)
    assert model.calls[0][2]["content"] == "FINAL ANSWER: A1"


def test_json_wrapped_code_step_is_executed(df):
    # Raw newline inside the JSON string, as the model actually sends it.
    reply = '{ "action": "python", "code": "total = df.revenue.sum()\nprint(total)" }'
    _, result = run(df, [reply, "FINAL ANSWER: Total is 100."])
    assert result.steps[0].output.strip() == "100"


def test_empty_answer_is_sent_back(df):
    model, result = run(df, ["", "FINAL ANSWER: East is weakest."])
    assert "no answer" in model.calls[1][-1]["content"]
    assert result.answer == "East is weakest."


def test_html_report_embeds_charts():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "**Bold** answer", "figures": [b"png-bytes"],
         "sources": [{"title": "Src", "url": "https://example.com"}]},
    ]
    html = build_html_report("data.csv", messages)
    assert "<strong>Bold</strong>" in html and "data:image/png;base64," in html and "https://example.com" in html
