"""Offline checks for grounded replies, memory commits and complete graph turns."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from agent import EvidenceContext, build_agent_graph, finalize
from agent.nodes import HISTORY_TOKEN_BUDGET, MAX_HISTORY_TURNS
from agent.response import RULEBOOK_CONTEXT_MAX_BYTES, SQL_CONTEXT_MAX_ROWS
from retrieval.rulebook_retriever import RetrievedChunk, RetrievedRulebookContext
from sql_retriever import RetrievalResult
from test_agent_graph import REFINED, rulebook_stub
from test_agent_nodes import planning_output
from test_sql_hybrid import client_for


def response_output(answer="Warboss belongs to Orks. [S1]", *, insufficient=False, memory=None):
    return {"answer": answer, "insufficient_evidence": insufficient, "memory": memory or []}


class AgentResponseTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_agent_graph()
        self.config = {"configurable": {"thread_id": "session-1"}}
        self.sql = Mock(return_value=RetrievalResult(
            "ok", "fixed", ["Id", "name", "factionId", "faction_name", "role"],
            [("000000001", "Warboss", "ORK", "Orks", "")],
            sql="SELECT Id, name, factionId, faction_name, role FROM metadata", parameters={"d_id": "000000001"},
        ))
        self.rules = rulebook_stub()
        self.context = EvidenceContext(rulebook_retriever=self.rules, sql_retriever=self.sql)

    def invoke(self, *outputs, question="Warboss 属于哪个阵营？", run_id="run-1", **state):
        self.context.client = client_for(*outputs)
        return self.graph.invoke({"user_input": question, "run_id": run_id, **state},
                                 self.config, context=self.context)

    def test_complete_routes_return_only_real_citations_and_commit_turns(self):
        for route, dependent in [("sql", False), ("rulebook", False), ("both", False), ("both", True)]:
            with self.subTest(route=route, dependent=dependent):
                self.setUp()
                markers = {"sql": "[S1]", "rulebook": "[R1]", "both": "[S1] [R1]"}[route]
                self.context.client = client_for(
                    planning_output(route, depends_on_sql=dependent),
                    *([REFINED] if dependent else []), response_output(f"有证据支持的回答。 {markers}"),
                )
                events = list(self.graph.stream(
                    {"user_input": "原始问题", "run_id": "run-1"}, self.config,
                    context=self.context, stream_mode="updates",
                ))
                self.assertEqual([next(iter(event)) for event in events][-2:], ["respond", "finalize"])
                result = self.graph.get_state(self.config).values
                self.assertEqual(result["outcome"], "completed")
                self.assertEqual([message.type for message in result["messages"]], ["human", "ai"])
                self.assertEqual(result["messages"][0].content, "原始问题")
                self.assertEqual(result["messages"][1].content, result["answer"])
                expected_refs = {"S1"} if route == "sql" else {"R1"} if route == "rulebook" else {"R1", "S1"}
                self.assertEqual(set(result["citations"]), expected_refs)
                if "S1" in expected_refs:
                    self.assertEqual(result["citations"]["S1"], {"source": "sql", "source_id": "run-1:sql"})
                    self.assertEqual(result["sql_result"].parameters["d_id"], "000000001")
                if "R1" in expected_refs:
                    self.assertEqual(result["citations"]["R1"], {"source": "rulebook", "source_id": "r-001"})
                payload = json.loads(self.context.client.models.generate_content.call_args.kwargs["contents"])
                self.assertEqual(payload["question"], "原始问题")
                self.assertNotIn("history", payload)
                if "R1" in expected_refs:
                    self.assertEqual(payload["sources"]["R1"]["pdf_page_numbers"], [7])
                self.assertEqual(self.context.client.models.generate_content.call_count, 3 if dependent else 2)

    def test_followup_recovers_committed_history_and_verified_identity(self):
        first = self.invoke(planning_output(), response_output(memory=[
            {"key": "unit", "value": "Warboss", "source": "sql"},
            {"key": "unit_id", "value": "000000001", "source": "sql"},
        ]))
        self.assertEqual(first["conversation_context"]["unit_id"], {"value": "000000001", "source_run_id": "run-1"})
        self.sql.return_value = RetrievalResult("ok", columns=["unit_name", "t"], rows=[("Warboss", "5")])
        second = self.invoke(planning_output(
            standalone_question="What is the Toughness of Warboss ID 000000001?",
            sql_question="Get Toughness for Warboss ID 000000001",
        ), response_output("Warboss 的 Toughness 为 5。[S1]"), question="它的 Toughness 呢？", run_id="run-2")
        calls = self.context.client.models.generate_content.call_args_list
        planning_payload = json.loads(calls[0].kwargs["contents"])
        self.assertEqual(planning_payload["history"], [
            {"role": "human", "content": first["user_input"]}, {"role": "ai", "content": first["answer"]},
        ])
        self.assertEqual(planning_payload["conversation_context"], first["conversation_context"])
        self.sql.assert_called_with("Get Toughness for Warboss ID 000000001")
        self.assertEqual(len(second["messages"]), 4)
        self.assertEqual(second["citations"]["S1"]["source_id"], "run-2:sql")
        self.assertEqual(second["conversation_context"], first["conversation_context"])
        self.assertEqual(self.graph.get_state(self.config).values, second)

    def test_clarification_displays_exact_ordered_candidates_then_continues_both(self):
        candidates = [{"id": "000000001", "name": "Boyz", "faction_id": "ORK"},
                      {"id": "000000002", "name": "Boyz", "faction_id": "TYR"}]
        self.sql.return_value = RetrievalResult("needs_clarification", message="Which Boyz?", candidates=candidates)
        first = self.invoke(planning_output("both", depends_on_sql=True), response_output("请选择一个单位。", insufficient=True))
        self.assertEqual(first["outcome"], "needs_clarification")
        for number, candidate in enumerate(candidates, 1):
            self.assertIn(f"{number}. {json.dumps(candidate, ensure_ascii=False)}", first["answer"])
        self.assertEqual(first["pending_clarification"]["candidates"], candidates)
        self.rules.retrieve.assert_not_called()
        self.sql.return_value = RetrievalResult("ok", columns=["datasheet_name", "ability_text"], rows=[("Boyz", "Feel No Pain")])
        second = self.invoke(planning_output(
            "both", depends_on_sql=True, sql_question="Boyz ID 000000002 abilities",
            standalone_question="Boyz ID 000000002 abilities and referenced Core Rules",
        ), REFINED, response_output("单位能力及规则解释。[S1] [R1]"), question="第二个", run_id="run-2")
        self.assertIsNone(second["pending_clarification"])
        self.assertEqual(second["outcome"], "completed")
        self.sql.assert_called_with("Boyz ID 000000002 abilities")
        self.assertEqual(len(second["messages"]), 4)

    def test_independent_partial_answers_preserve_missing_source_status(self):
        for status in ("empty", "error", "unsupported"):
            with self.subTest(status=status):
                self.setUp()
                self.sql.return_value = RetrievalResult(status, message="Database part unavailable.")
                result = self.invoke(planning_output("both"), response_output("只能解释规则，数据库部分未完成。[R1]", insufficient=True))
                self.assertEqual(result["outcome"], "partial")
                self.assertEqual(set(result["citations"]), {"R1"})
                payload = json.loads(self.context.client.models.generate_content.call_args.kwargs["contents"])
                self.assertEqual(payload["sql_status"], status)
                self.assertNotIn("S1", payload["sources"])
                self.assertEqual(len(result["messages"]), 2)

    def test_empty_error_and_nonretrieval_responses_have_distinct_outcomes(self):
        for route, sql_status, outcome in [
            ("sql", "empty", "partial"), ("sql", "error", "error"), ("sql", "unsupported", "unsupported"),
            ("direct", None, "completed"), ("clarify", None, "needs_clarification"), ("unsupported", None, "unsupported"),
        ]:
            with self.subTest(route=route, sql_status=sql_status):
                self.setUp()
                if sql_status:
                    self.sql.return_value = RetrievalResult(sql_status, message="Source-specific explanation")
                result = self.invoke(planning_output(route), response_output("当前请求的说明。", insufficient=True))
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["citations"], {})
                self.assertEqual(len(result["messages"]), 0 if outcome == "error" else 2)
                if route != "sql":
                    self.sql.assert_not_called()
                self.rules.retrieve.assert_not_called()
                payload = json.loads(self.context.client.models.generate_content.call_args.kwargs["contents"])
                self.assertEqual(payload["sources"], {})

    def test_invalid_answer_or_provider_failure_never_commits_or_retries(self):
        outputs = [response_output("Made up source. [R99]"), response_output("Uncited fact."),
                   response_output("Malformed reference [S1, S2] alongside [S1]."),
                   response_output("Invalid SQL source. [S2]"), {"answer": "", "insufficient_evidence": False, "memory": []},
                   {"answer": "Fact. [S1]", "insufficient_evidence": "false", "memory": []},
                   RuntimeError("Provider failed")]
        for output in outputs:
            with self.subTest(output=output):
                self.setUp()
                self.context.client = client_for(planning_output(), output if isinstance(output, dict) else {})
                if isinstance(output, Exception):
                    self.context.client.models.generate_content.side_effect = [
                        SimpleNamespace(text=json.dumps(planning_output())), output,
                    ]
                previous = [HumanMessage(content="old question"), AIMessage(content="old answer")]
                with self.assertRaises((ValueError, RuntimeError)):
                    self.graph.invoke({"user_input": "Warboss?", "run_id": "run-1", "messages": previous},
                                      self.config, context=self.context)
                restored = self.graph.get_state(self.config).values
                self.assertEqual(restored["messages"], previous)
                self.assertIsNone(restored["answer"])
                self.assertIsNone(restored["outcome"])
                self.assertEqual(restored["citations"], {})
                self.assertEqual(self.context.client.models.generate_content.call_count, 2)

    def test_unverified_or_conflicting_memory_is_rejected(self):
        invalid = [
            [{"key": "unit_id", "value": "1", "source": "sql"}],
            [{"key": "unit", "value": "Invented unit", "source": "sql"}],
            [{"key": "conditions", "value": "Toughness >= 10", "source": "user"}],
            [{"key": "toughness", "value": "5", "source": "sql"}],
            [{"key": "unit", "value": "Warboss", "source": "user"}] * 2,
        ]
        for memory in invalid:
            with self.subTest(memory=memory):
                self.setUp()
                with self.assertRaises(ValueError):
                    self.invoke(planning_output(), response_output(memory=memory))
                self.assertEqual(self.graph.get_state(self.config).values["messages"], [])

    def test_user_conditions_are_copied_verbatim_and_new_topic_clears_old_scope(self):
        first = self.invoke(planning_output(), response_output(memory=[
            {"key": "faction", "value": "Orks", "source": "user"},
            {"key": "conditions", "value": "Toughness >= 10，排除 Warboss", "source": "user"},
        ]), question="Orks，Toughness >= 10，排除 Warboss")
        self.assertEqual(first["conversation_context"]["conditions"]["value"], "Toughness >= 10，排除 Warboss")
        second = self.invoke(planning_output("rulebook", starts_new_topic=True),
                             response_output("Advance 的规则解释。[R1]"), question="换个话题，解释 Advance", run_id="run-2")
        self.assertEqual(second["conversation_context"], {})
        self.assertIsNone(second["sql_result"])
        self.assertEqual(second["context_updates"], {})

    def test_changing_identity_clears_the_old_name_or_id_partner(self):
        for changed_key, partner, value in [("unit", "unit_id", "Boyz"), ("unit_id", "unit", "000000002")]:
            with self.subTest(changed_key=changed_key):
                self.setUp()
                self.sql.return_value = RetrievalResult("ok", columns=["unit_name"], rows=[("Boyz",)])
                result = self.invoke(planning_output(), response_output(memory=[
                    {"key": changed_key, "value": value, "source": "user"},
                ]), question=f"Now show {value}", conversation_context={
                    "unit": {"value": "Warboss", "source_run_id": "old-run"},
                    "unit_id": {"value": "000000001", "source_run_id": "old-run"},
                })
                self.assertEqual(result["conversation_context"][changed_key]["value"], value)
                self.assertNotIn(partner, result["conversation_context"])

    def test_sql_lists_and_truncated_results_do_not_offer_arbitrary_active_entities(self):
        for rows, truncated in [([("Warboss",), ("Boyz",)], False), ([("Warboss",)], True)]:
            with self.subTest(truncated=truncated):
                self.setUp()
                self.sql.return_value = RetrievalResult("ok", columns=["unit_name"], rows=rows, truncated=truncated)
                self.invoke(planning_output(), response_output())
                payload = json.loads(self.context.client.models.generate_content.call_args.kwargs["contents"])
                self.assertEqual(payload["sql_memory"], {})

    def test_answer_context_clipping_keeps_original_evidence_and_forces_partial(self):
        self.sql.return_value = RetrievalResult("ok", columns=["id", "name"],
                                               rows=[(str(number), "unit") for number in range(60)])
        result = self.invoke(planning_output(), response_output("结果已裁剪，以下仅为部分数据。[S1]"))
        payload = json.loads(self.context.client.models.generate_content.call_args.kwargs["contents"])
        self.assertEqual(len(payload["sources"]["S1"]["rows"]), SQL_CONTEXT_MAX_ROWS)
        self.assertEqual(len(result["sql_result"].rows), 60)
        self.assertFalse(result["sql_context_truncated"])
        self.assertTrue(result["answer_context_truncated"]["sql"])
        self.assertEqual(result["evidence_status"], "ok")
        self.assertEqual(result["outcome"], "partial")

        self.rules.retrieve.side_effect = lambda query: RetrievedRulebookContext(query, (
            RetrievedChunk({"chunk_id": "too-large", "text": "x" * RULEBOOK_CONTEXT_MAX_BYTES}, 1, 1, 1, 0.05, 0.9),
        ))
        no_context = self.invoke(planning_output("rulebook"), response_output("规则证据超出上下文预算。", insufficient=True), run_id="run-2")
        self.assertTrue(no_context["answer_context_truncated"]["rulebook"])
        self.assertTrue(no_context["rulebook_result"].results)
        self.assertEqual(no_context["citations"], {})
        self.assertEqual(no_context["outcome"], "partial")

    def test_missing_source_citations_or_insufficient_evidence_cannot_claim_completion(self):
        for output in [response_output("Only rulebook facts. [R1]"),
                       response_output("Evidence does not answer this question.", insufficient=True)]:
            with self.subTest(output=output):
                self.setUp()
                result = self.invoke(planning_output("both"), output)
                self.assertEqual(result["outcome"], "partial")

    def test_finalize_is_idempotent_and_ignores_failed_or_cancelled_drafts(self):
        state = self.invoke(planning_output(), response_output())
        update = finalize(state)
        self.assertEqual(update["messages"].value, state["messages"])
        self.assertEqual([message.id for message in state["messages"]], ["run-1:user", "run-1:assistant"])
        for outcome in (None, "error", "cancelled"):
            self.assertEqual(finalize({**state, "outcome": outcome, "answer": "Incomplete draft"}), {})
        self.assertEqual(finalize({**state, "answer": None}), {})

    def test_history_stays_bounded_while_user_scope_survives_trimming(self):
        for number in range(10):
            memory = [{"key": "faction", "value": "Orks", "source": "user"}] if number == 0 else []
            result = self.invoke(planning_output(), response_output(memory=memory),
                                 question=f"Orks question {number}", run_id=f"run-{number}")
        self.assertEqual(len(result["messages"]), 2 * MAX_HISTORY_TURNS)
        self.assertEqual(result["messages"][0].id, "run-2:user")
        self.assertEqual(result["conversation_context"]["faction"], {"value": "Orks", "source_run_id": "run-0"})
        self.assertLessEqual(count_tokens_approximately(result["messages"]), HISTORY_TOKEN_BUDGET)


if __name__ == "__main__":
    unittest.main()
