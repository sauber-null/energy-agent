"""The agent: a plain tool-calling loop.

1. Send the conversation and the tool schemas to the model.
2. If the model answers with tool calls, run them and append their results.
3. Repeat until the model answers with text or the step budget is used up. The last
   step offers no tools, which forces a final answer.

Guardrail: a prompt can ask the model to use tools but cannot enforce it, so the loop
checks it. An answer given before any successful tool call is rejected once with a
reminder; if the model still answers without tool results, the answer is returned with
grounded=False so callers can flag or withhold it.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from energy_agent.llm import ChatModel, Message
from energy_agent.tools.registry import Tool
from energy_agent.tracing import Tracer

SYSTEM_PROMPT = """\
You are an assistant for German electricity market processes: the GPKE and the BNetzA \
decision BK6-22-024 (supplier switch within 24 hours, in force since 2025-06-06). \
Today is {today}.

Rules:
- Base every statement on tool results. Use search_documents for rules, roles and definitions.
- Never compute dates or deadlines yourself; call calculate_deadline.
- For a concrete switch request (request ID or MaLo-ID) call get_switch_status first.
- Name the source (document, section, page) for each rule or deadline you state.
- If the tools do not answer the question, say so instead of guessing.
- Answer in the language of the question, briefly and precisely."""

TOOL_REMINDER = (
    "You answered without using a tool. Do not answer from memory: call search_documents, "
    "calculate_deadline or get_switch_status first and base the answer on the result."
)

StopReason = Literal["answer", "max_steps"]


@dataclass
class AgentResult:
    question: str
    answer: str
    run_id: str
    stop_reason: StopReason
    llm_calls: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    guardrail_retries: int = 0

    @property
    def grounded(self) -> bool:
        """At least one tool call returned a result (not an error)."""
        return any("error" not in call["result"] for call in self.tool_calls)


class Agent:
    def __init__(
        self,
        model: ChatModel,
        tools: list[Tool],
        tracer: Tracer | None = None,
        max_steps: int = 6,
        today: date | None = None,
        require_tool_use: bool = True,
    ):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.schemas = [tool.schema() for tool in tools]
        self.tracer = tracer or Tracer(path=None)
        self.max_steps = max_steps
        self.today = today or date.today()
        self.require_tool_use = require_tool_use

    def run(self, question: str) -> AgentResult:
        run_id = uuid.uuid4().hex[:12]
        result = AgentResult(question, "", run_id, "max_steps")
        messages = [
            Message("system", SYSTEM_PROMPT.format(today=self.today.isoformat())),
            Message("user", question),
        ]
        self.tracer.event(
            run_id,
            "run_start",
            question=question,
            provider=self.model.provider,
            model=self.model.model,
        )

        for step in range(1, self.max_steps + 1):
            last_step = step == self.max_steps
            response = self.model.chat(messages, [] if last_step else self.schemas)
            reply = response.message
            messages.append(reply)

            result.llm_calls += 1
            result.prompt_tokens += response.prompt_tokens
            result.completion_tokens += response.completion_tokens
            result.latency_ms += response.latency_ms
            self.tracer.event(
                run_id,
                "llm_call",
                step=step,
                model=response.model,
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                tool_calls=[c.name for c in reply.tool_calls],
                content=reply.content,
            )

            if not reply.tool_calls:
                if (
                    self.require_tool_use
                    and not result.grounded
                    and result.guardrail_retries == 0
                    and not last_step
                ):
                    result.guardrail_retries += 1
                    self.tracer.event(run_id, "guardrail", step=step, rule="answer_without_tool")
                    # A user message, because not every provider accepts a system message
                    # in the middle of a conversation.
                    messages.append(Message("user", TOOL_REMINDER))
                    continue
                result.answer = reply.content
                result.stop_reason = "max_steps" if last_step else "answer"
                break

            for call in reply.tool_calls:
                output = self._run_tool(call.name, call.arguments)
                result.tool_calls.append(
                    {"name": call.name, "arguments": call.arguments, "result": output}
                )
                self.tracer.event(
                    run_id,
                    "tool_call",
                    step=step,
                    tool=call.name,
                    arguments=call.arguments,
                    error=output.get("error"),
                    result=output,
                )
                messages.append(
                    Message(
                        "tool",
                        json.dumps(output, ensure_ascii=False, default=str),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        self.tracer.event(
            run_id,
            "run_end",
            stop_reason=result.stop_reason,
            grounded=result.grounded,
            guardrail_retries=result.guardrail_retries,
            answer=result.answer,
            llm_calls=result.llm_calls,
            tool_calls=len(result.tool_calls),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
        )
        return result

    def _run_tool(self, name: str, arguments: dict | str) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'. Available: {', '.join(self.tools)}"}
        return tool.run(arguments)
