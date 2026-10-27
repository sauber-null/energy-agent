from collections.abc import Sequence
from datetime import date

from energy_agent.agent import Agent
from energy_agent.llm import ChatResponse, Message, ToolCall
from energy_agent.tools.registry import build_tools
from energy_agent.tracing import Tracer


class ScriptedModel:
    """Returns prepared replies in order and records what it was sent."""

    provider = "scripted"
    model = "scripted-1"

    def __init__(self, replies: list[Message]):
        self.replies = list(replies)
        self.calls: list[tuple[list[Message], list[dict]]] = []

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> ChatResponse:
        self.calls.append((list(messages), list(tools)))
        return ChatResponse(self.replies.pop(0), self.model, 100, 10, 5)


def make_agent(replies: list[Message], max_steps: int = 6) -> tuple[Agent, ScriptedModel]:
    model = ScriptedModel(replies)
    tools = build_tools(conn=None, retriever=None)
    return Agent(model, tools, Tracer(path=None), max_steps, date(2026, 10, 27)), model


DEADLINE_CALL = ToolCall(
    "c1",
    "calculate_deadline",
    {"process": "lieferbeginn", "step": 6, "step1_sent_at": "2026-12-23T10:00"},
)


def test_tool_result_is_sent_back_and_final_answer_returned():
    agent, model = make_agent(
        [
            Message("assistant", tool_calls=(DEADLINE_CALL,)),
            Message("assistant", "Bis Mo 28.12.2026, 11:00 Uhr (GPKE Teil 2, Nr. 6)."),
        ]
    )
    result = agent.run("Bis wann muss der NB ablehnen?")

    assert result.answer.startswith("Bis Mo 28.12.2026")
    assert result.stop_reason == "answer"
    assert result.llm_calls == 2
    assert result.tool_calls[0]["result"]["deadline"] == "2026-12-28T11:00"

    second_call_messages = model.calls[1][0]
    tool_message = second_call_messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "c1"
    assert "2026-12-28T11:00" in tool_message.content
    assert "2026-10-27" in second_call_messages[0].content  # today in the system prompt


def test_unknown_tool_and_bad_arguments_become_errors_for_the_model():
    agent, model = make_agent(
        [
            Message(
                "assistant",
                tool_calls=(
                    ToolCall("c1", "delete_database", {}),
                    ToolCall("c2", "calculate_deadline", "{broken json"),
                ),
            ),
            Message("assistant", "Das kann ich nicht beantworten."),
        ]
    )
    result = agent.run("?")
    errors = [call["result"]["error"] for call in result.tool_calls]
    assert "Unknown tool 'delete_database'" in errors[0]
    assert errors[1]
    assert result.stop_reason == "answer"


def test_last_step_offers_no_tools_to_force_an_answer():
    looping = Message("assistant", tool_calls=(DEADLINE_CALL,))
    agent, model = make_agent([looping, looping, Message("assistant", "Antwort")], max_steps=3)
    result = agent.run("?")
    assert [len(tools) for _, tools in model.calls] == [3, 3, 0]
    assert result.answer == "Antwort"
    assert result.stop_reason == "max_steps"


def test_every_step_is_traced():
    agent, _ = make_agent(
        [Message("assistant", tool_calls=(DEADLINE_CALL,)), Message("assistant", "ok")]
    )
    agent.run("?")
    events = [e["event"] for e in agent.tracer.events]
    assert events == ["run_start", "llm_call", "tool_call", "llm_call", "run_end"]
    assert len({e["run_id"] for e in agent.tracer.events}) == 1
