"""Offline checks for turn isolation and structured, single-call planning."""

import json
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import END, START, StateGraph

from agent import AgentState, plan, prepare_turn
from agent.nodes import HISTORY_TOKEN_BUDGET, MAX_HISTORY_TURNS
from retrieval.rulebook_retriever import RetrievedRulebookContext, RulebookQuerySet
from sql_retriever.result import RetrievalResult


def planning_output(route="sql", **changes):
    rules_needed = route in ("rulebook", "both")
    return {
        "standalone_question": "What is the Toughness of Warboss?",
        "route": route,
        "reason": "Retrieve the requested evidence.",
        "sql_question": "What is the Toughness of Warboss?" if route in ("sql", "both") else None,
        "semantic_query": "How does an Advance move work?" if rules_needed else None,
        "keyword_query": ["Advance", "Normal Move", "Movement phase"] if rules_needed else None,
        "depends_on_sql": False,
        "message": "Please specify the request." if route in ("clarify", "unsupported") else "",
        "starts_new_topic": False,
        **changes,
    }


def client_for(output):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(output))
    return client


class AgentNodeTests(unittest.TestCase):
    def test_prepare_clears_old_evidence_and_preserves_conversation(self):
        messages = [HumanMessage(content="Warboss abilities"), AIMessage(content="Abilities.")]
        pending = {
            "question": "Show Boyz stats", "message": "Which Boyz?",
            "candidates": [{"id": "000000001", "name": "Boyz"}], "source_run_id": "run-1",
        }
        context = {"faction": {"value": "Orks", "source_run_id": "run-1"}}
        old_result = RetrievalResult("ok", rows=[("old data",)])
        state = AgentState(
            messages=messages, conversation_context=context, pending_clarification=pending,
            run_id="run-2", turn_index=1, user_input="The first one",
            standalone_question="Old question",
            route_plan={
                "route": "sql", "reason": "Old plan", "sql_question": "Old question",
                "rulebook_query_set": None, "depends_on_sql": False, "message": "",
            },
            sql_result=old_result,
            rulebook_result=RetrievedRulebookContext(RulebookQuerySet("Old rule question"), ()),
            evidence_status="partial", retrieval_errors={"rulebook": "OSError"}, sql_context_truncated=True,
            answer_context_truncated={"sql": True},
            context_updates={"unit": {"value": "Old unit", "source_run_id": "run-1"}},
            answer="Old answer", citations={"S1": {"source": "sql", "source_id": "run-1:sql"}},
            outcome="completed",
        )
        graph = StateGraph(AgentState)
        graph.add_node("prepare", prepare_turn)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", END)
        updated = graph.compile().invoke(state)

        self.assertEqual(updated["turn_index"], 2)
        self.assertEqual(updated["run_id"], "run-2")
        self.assertEqual(updated["messages"], messages)
        self.assertEqual(updated["pending_clarification"], pending)
        self.assertEqual(updated["conversation_context"], context)
        for field in ("standalone_question", "route_plan", "sql_result", "rulebook_result", "evidence_status", "answer", "outcome"):
            self.assertIsNone(updated[field])
        self.assertEqual(updated["citations"], {})
        self.assertEqual(updated["retrieval_errors"], {})
        self.assertFalse(updated["sql_context_truncated"])
        self.assertEqual(updated["answer_context_truncated"], {})
        self.assertEqual(updated["context_updates"], {})
        self.assertIs(state["sql_result"], old_result)
        self.assertEqual(state["answer"], "Old answer")

    def test_history_window_replaces_messages_and_keeps_complete_recent_turns(self):
        messages = []
        for number in range(10):
            messages.extend([
                HumanMessage(content=f"Question {number}", id=f"user-{number}"),
                AIMessage(content=f"Answer {number}", id=f"assistant-{number}"),
            ])
        client = client_for(planning_output())
        graph = StateGraph(AgentState)
        graph.add_node("prepare", prepare_turn)
        graph.add_node("plan", partial(plan, client=client))
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "plan")
        graph.add_edge("plan", END)
        result = graph.compile().invoke({
            "messages": messages, "run_id": "new-run", "user_input": "Its Toughness?",
        })

        self.assertEqual(len(result["messages"]), 2 * MAX_HISTORY_TURNS)
        self.assertEqual(result["messages"][0].id, "user-2")
        payload = json.loads(client.models.generate_content.call_args.kwargs["contents"])
        self.assertEqual(payload["history"][0]["content"], "Question 2")
        self.assertEqual(payload["question"], "Its Toughness?")
        self.assertNotIn("Its Toughness?", [item["content"] for item in payload["history"]])
        self.assertEqual(result["turn_index"], 1)
        self.assertEqual(result["route_plan"]["route"], "sql")

    def test_large_old_turn_is_removed_without_splitting_messages(self):
        messages = [
            HumanMessage(content="old " * 25_000), AIMessage(content="Old answer"),
            HumanMessage(content="Warboss?"), AIMessage(content="Which statistic?"),
        ]
        history = prepare_turn({"messages": messages})["messages"].value
        self.assertEqual([message.content for message in history], ["Warboss?", "Which statistic?"])
        self.assertLessEqual(count_tokens_approximately(history), HISTORY_TOKEN_BUDGET)

    def test_rulebook_query_uses_exact_original_input(self):
        question = "  Advance 后能做什么？  "
        client = client_for(planning_output("rulebook"))
        result = plan({"user_input": question}, client=client)
        query_set = result["route_plan"]["rulebook_query_set"]

        self.assertIsInstance(query_set, RulebookQuerySet)
        self.assertEqual(query_set.original_query, question)
        self.assertEqual(query_set.keyword_query, ("Advance", "Normal Move", "Movement phase"))
        self.assertIsNone(result["route_plan"]["sql_question"])
        client.models.generate_content.assert_called_once()

    def test_sql_route_uses_the_full_question_instead_of_a_second_model_written_query(self):
        question = "What is the Toughness of an Imperial Space Marine?"
        client = client_for(planning_output(
            standalone_question=question,
            sql_question="SELECT toughness FROM units WHERE name='Imperial Space Marine' OR name LIKE '%Space Marine%'",
        ))
        result = plan({"user_input": question}, client=client)
        self.assertEqual(result["route_plan"]["sql_question"], question)
        client.models.generate_content.assert_called_once()

    def test_followup_passes_history_conditions_and_exact_candidates_once(self):
        pending = {
            "question": "Show Boyz stats", "message": "Which unit?", "source_run_id": "run-1",
            "candidates": [
                {"id": "000000001", "name": "Boyz", "faction_id": "ORK"},
                {"id": "000000002", "name": "Boyz", "faction_id": "TYR"},
            ],
        }
        context = {"faction": {"value": "Tyranids", "source_run_id": "run-1"}}
        client = client_for(planning_output(
            standalone_question="Show stats for Boyz with ID 000000002",
            sql_question="Show stats for Boyz with ID 000000002",
        ))
        state = {
            "user_input": "第二个", "pending_clarification": pending, "conversation_context": context,
            "messages": [HumanMessage(content="Show Boyz stats"), AIMessage(content="Which unit?")],
        }
        result = plan(state, client=client)
        payload = json.loads(client.models.generate_content.call_args.kwargs["contents"])

        self.assertEqual(payload["pending_clarification"], pending)
        self.assertEqual(payload["conversation_context"], context)
        self.assertEqual(len(payload["history"]), 2)
        self.assertIn("000000002", result["route_plan"]["sql_question"])
        self.assertNotIn("pending_clarification", result)
        self.assertNotIn("messages", result)
        client.models.generate_content.assert_called_once()

    def test_both_modes_are_selected_by_the_same_planning_call(self):
        for dependent in (False, True):
            with self.subTest(depends_on_sql=dependent):
                client = client_for(planning_output("both", depends_on_sql=dependent))
                result = plan({"user_input": "Combine unit data and rules"}, client=client)
                route = result["route_plan"]
                self.assertEqual(route["route"], "both")
                self.assertIs(route["depends_on_sql"], dependent)
                self.assertIsNotNone(route["sql_question"])
                self.assertIsInstance(route["rulebook_query_set"], RulebookQuerySet)
                client.models.generate_content.assert_called_once()

    def test_new_topic_clears_previous_scope_and_pending_question(self):
        client = client_for(planning_output("rulebook", starts_new_topic=True))
        result = plan({
            "user_input": "Explain Advance",
            "conversation_context": {"unit": {"value": "Warboss", "source_run_id": "run-1"}},
            "pending_clarification": {
                "question": "Show Boyz stats", "message": "Which Boyz?", "candidates": [],
                "source_run_id": "run-1",
            },
        }, client=client)
        self.assertEqual(result["conversation_context"], {})
        self.assertIsNone(result["pending_clarification"])

    def test_non_retrieval_routes_have_no_queries_or_assistant_message(self):
        for route in ("direct", "clarify", "unsupported"):
            with self.subTest(route=route):
                result = plan({"user_input": "Help"}, client=client_for(planning_output(route)))
                self.assertIsNone(result["route_plan"]["sql_question"])
                self.assertIsNone(result["route_plan"]["rulebook_query_set"])
                self.assertNotIn("messages", result)
                self.assertNotIn("answer", result)

    def test_invalid_model_output_stops_without_retry_or_route_fallback(self):
        invalid = [
            planning_output(route="unknown"),
            planning_output(sql_question=None),
            planning_output(depends_on_sql=True),
            planning_output(depends_on_sql="false"),
            planning_output("rulebook", keyword_query=["Advance"]),
            planning_output("rulebook", semantic_query="   "),
            planning_output("direct", sql_question="Get Warboss"),
            planning_output("clarify", message=""),
            planning_output(unexpected="extra field"),
        ]
        for output in invalid:
            with self.subTest(output=output):
                client = client_for(output)
                with self.assertRaises(ValueError):
                    plan({"user_input": "Question"}, client=client)
                client.models.generate_content.assert_called_once()
        client = client_for({})
        client.models.generate_content.return_value.text = "not JSON"
        with self.assertRaises(ValueError):
            plan({"user_input": "Question"}, client=client)
        client.models.generate_content.assert_called_once()

    def test_provider_failure_propagates_without_retry(self):
        client = client_for(planning_output())
        client.models.generate_content.side_effect = RuntimeError("Provider unavailable")
        with self.assertRaisesRegex(RuntimeError, "Provider unavailable"):
            plan({"user_input": "Question"}, client=client)
        client.models.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
