"""Test vòng lặp agent với LLM giả (không gọi API). Không dùng để báo chất lượng."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import ABSTAIN_TEXT, Agent, AgentConfig, prepare_input  # noqa: E402

CHUNKS = [
    {"id": "B:doc:p026:004:aa", "text": "A grocery chain said diesel prices rose thirty percent.",
     "source_id": "doc", "source": "doc.pdf", "page": 26, "metadata": {"district": "Richmond", "section": "Retail"}},
    {"id": "B:doc:p048:001:bb", "text": "Diesel was the top cost driver.",
     "source_id": "doc", "source": "doc.pdf", "page": 48, "metadata": {"district": "Dallas", "section": "Prices"}},
]

CASE = {"id": "q09", "split": "dev", "question": "How much did diesel rise for a Richmond grocer?",
        "filter": None, "answerable": True,
        "expected_answer": "SECRET_EXPECTED_ANSWER_42",
        "gold": [{"source_id": "doc", "page": 26, "evidence": "SECRET_GOLD_EVIDENCE_99"}],
        "grounding_requirements": ["SECRET_REQUIREMENT"]}


def tool_call(name, args, i=0):
    return NS(id=f"call{i}", function=NS(name=name, arguments=json.dumps(args)))


class FakeLLM:
    """Trả lần lượt các tool call đã lên kịch bản; ghi lại mọi request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw, default=str)))
        name, args = self.script.pop(0)
        msg = NS(content=None, tool_calls=[tool_call(name, args, len(self.requests))])
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=100, completion_tokens=10))


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, filters=None, top_k=5):
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        hits = [c for c in CHUNKS if not filters or c["metadata"]["district"] == filters["district"]]
        return [{**c, "score": 1.0, "rank": i} for i, c in enumerate(hits[:top_k], 1)]


def make(script, **cfg):
    llm, ret = FakeLLM(script), FakeRetriever()
    agent = Agent(llm, ret, AgentConfig(model="fake", **cfg), ["Dallas", "Richmond"])
    return agent, llm, ret


class AgentTests(unittest.TestCase):
    def test_prepare_input_keeps_only_allowed_fields(self):
        self.assertEqual(set(prepare_input(CASE)), {"id", "question", "filter"})

    def test_gold_never_reaches_llm_or_retriever(self):
        agent, llm, ret = make([("search", {"query": "Richmond grocer diesel"}),
                                ("submit_answer", {"answer": "Up 30% [S1].", "citations": ["S1"], "abstain": False})])
        agent.run(CASE)
        sent = json.dumps(llm.requests) + json.dumps(ret.calls)
        for secret in ("SECRET_EXPECTED_ANSWER_42", "SECRET_GOLD_EVIDENCE_99", "SECRET_REQUIREMENT"):
            self.assertNotIn(secret, sent)

    def test_answer_with_valid_citation_maps_label_to_chunk_id(self):
        agent, _, _ = make([("search", {"query": "diesel"}),
                            ("submit_answer", {"answer": "Up 30% [S1].", "citations": ["S1"], "abstain": False})])
        tr = agent.run(CASE)
        self.assertEqual(tr["status"], "answered")
        self.assertEqual(tr["citations"], ["B:doc:p026:004:aa"])
        self.assertIn("[B:doc:p026:004:aa]", tr["answer"])
        self.assertEqual(tr["num_searches"], 1)
        self.assertEqual([e["event"] for e in tr["events"]], ["search", "submit_answer"])

    def test_abstain_is_recorded(self):
        agent, _, _ = make([("search", {"query": "unemployment rate"}),
                            ("submit_answer", {"answer": "Not stated.", "citations": [], "abstain": True})])
        tr = agent.run(CASE)
        self.assertEqual((tr["status"], tr["citations"]), ("abstain", []))

    def test_invalid_citation_gets_one_repair_then_abstains(self):
        bad = ("submit_answer", {"answer": "Up 30% [S9].", "citations": ["S9"], "abstain": False})
        agent, _, _ = make([("search", {"query": "diesel"}), bad, bad])
        tr = agent.run(CASE)
        self.assertEqual(tr["status"], "abstain")
        self.assertEqual(tr["answer"], ABSTAIN_TEXT)
        self.assertEqual(tr["note"], "invalid_or_missing_citations")
        self.assertIn("citation_repair_requested", [e["event"] for e in tr["events"]])

    def test_repair_can_succeed(self):
        agent, _, _ = make([("search", {"query": "diesel"}),
                            ("submit_answer", {"answer": "Up [S9].", "citations": ["S9"], "abstain": False}),
                            ("submit_answer", {"answer": "Up 30% [S1].", "citations": ["S1"], "abstain": False})])
        self.assertEqual(agent.run(CASE)["status"], "answered")

    def test_search_limit_forces_answer(self):
        script = [("search", {"query": f"q{i}"}) for i in range(2)]
        script.append(("submit_answer", {"answer": "Up 30% [S1].", "citations": ["S1"], "abstain": False}))
        agent, llm, _ = make(script, max_searches=2)
        tr = agent.run(CASE)
        self.assertEqual(tr["num_searches"], 2)
        self.assertIn("search_limit", [e["event"] for e in tr["events"]])
        forced = llm.requests[-1]["tool_choice"]
        self.assertEqual(forced["function"]["name"], "submit_answer")

    def test_user_filter_is_enforced_over_model_choice(self):
        agent, _, ret = make([("search", {"query": "diesel", "district": "Dallas"}),
                              ("submit_answer", {"answer": "Up 30% [S1].", "citations": ["S1"], "abstain": False})])
        tr = agent.run({**CASE, "filter": {"district": "Richmond"}})
        self.assertEqual(ret.calls[0]["filters"], {"district": "Richmond"})
        self.assertEqual(tr["events"][0]["filter_source"], "user")

    def test_model_district_used_when_named_in_question(self):
        agent, _, ret = make([("search", {"query": "diesel", "district": "Richmond"}),
                              ("submit_answer", {"answer": "Up [S1].", "citations": ["S1"], "abstain": False})])
        tr = agent.run(CASE)  # câu hỏi có chữ "Richmond"
        self.assertEqual(ret.calls[0]["filters"], {"district": "Richmond"})
        self.assertEqual(tr["events"][0]["filter_source"], "model")

    def test_model_district_rejected_when_not_in_question(self):
        agent, _, ret = make([("search", {"query": "diesel", "district": "Dallas"}),
                              ("submit_answer", {"answer": "Up [S1].", "citations": ["S1"], "abstain": False})])
        tr = agent.run(CASE)  # câu hỏi không nhắc Dallas
        self.assertIsNone(ret.calls[0]["filters"])
        self.assertEqual(tr["events"][0]["filter_source"], "model_rejected")

    def test_district_guard_can_be_disabled(self):
        agent, _, ret = make([("search", {"query": "diesel", "district": "Dallas"}),
                              ("submit_answer", {"answer": "Up [S1].", "citations": ["S1"], "abstain": False})],
                             district_guard=False)
        agent.run(CASE)  # thử nghiệm B: áp dụng district model đề xuất dù không có trong câu hỏi
        self.assertEqual(ret.calls[0]["filters"], {"district": "Dallas"})

    def test_prompt_v2_is_sent(self):
        agent, llm, _ = make([("search", {"query": "diesel"}),
                              ("submit_answer", {"answer": "Up [S1].", "citations": ["S1"], "abstain": False})],
                             prompt_version="v2")
        agent.run(CASE)
        self.assertIn("Do not add interpretation", llm.requests[0]["messages"][0]["content"])

    def test_citation_appended_when_missing_inline(self):
        agent, _, _ = make([("search", {"query": "diesel"}),
                            ("submit_answer", {"answer": "Up 30%.", "citations": ["S1"], "abstain": False})])
        tr = agent.run(CASE)
        self.assertTrue(tr["answer"].endswith("[B:doc:p026:004:aa]"))

    def test_duplicate_chunks_not_repeated_and_budget_respected(self):
        agent, llm, _ = make([("search", {"query": "a"}), ("search", {"query": "b"}),
                              ("submit_answer", {"answer": "x [S1].", "citations": ["S1"], "abstain": False})],
                             context_budget=60)
        tr = agent.run(CASE)
        self.assertEqual(tr["context_ids"], ["B:doc:p026:004:aa"])          # chunk 2 vượt budget
        self.assertEqual(tr["events"][0]["dropped_by_budget"], ["B:doc:p048:001:bb"])
        second_tool_msg = llm.requests[2]["messages"][-1]["content"]
        self.assertIn("already shown", second_tool_msg)

    def test_provider_error_is_reported_not_faked(self):
        agent, llm, _ = make([])
        def boom(**kw):
            raise ConnectionError("network down")
        llm.chat.completions.create = boom
        tr = agent.run(CASE)
        self.assertEqual(tr["status"], "error")
        self.assertIn("network down", tr["error"])


if __name__ == "__main__":
    unittest.main()
