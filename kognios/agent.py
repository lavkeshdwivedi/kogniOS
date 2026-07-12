from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar

from .guardrails.base import GuardrailError  # noqa: F401 – re-exported for convenience
from .models.base import BaseModel
from .tools.registry import ToolRegistry
from .tracing import Tracer

T = TypeVar("T")


class Agent:
    def __init__(
        self,
        model: BaseModel,
        tools: list[Callable] | None = None,
        memory=None,
        knowledge=None,
        instructions: str = "",
        max_iterations: int = 10,
        name: str = "",
        description: str = "",
        input_guardrails: list[Callable] | None = None,
        output_guardrails: list[Callable] | None = None,
        tracer: Tracer | None = None,
        tool_timeout: float = 30.0,
        max_context_tokens: int | None = None,
        max_tool_errors: int = 3,
    ):
        self.model = model
        self.registry = ToolRegistry(tools or [])
        self.memory = memory
        self.knowledge = knowledge
        self.instructions = instructions
        self.max_iterations = max_iterations
        self.name = name
        self.description = description
        self.last_usage: dict[str, int] = {}
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.tracer = tracer
        self.tool_timeout = tool_timeout
        self.max_context_tokens = max_context_tokens
        self.max_tool_errors = max_tool_errors

    # ── Guardrail helpers ────────────────────────────────────────────────────

    def _apply_input_guardrails(self, text: str) -> str:
        for guard in self.input_guardrails:
            text = guard(text)
        return text

    def _apply_output_guardrails(self, text: str) -> str:
        for guard in self.output_guardrails:
            text = guard(text)
        return text

    # ── Context trimming ─────────────────────────────────────────────────────

    def _trim_messages(self, messages: list[dict]) -> list[dict]:
        if self.max_context_tokens is None:
            return messages
        # ~4 chars per token; always keep the last message (current user turn)
        while len(messages) > 1:
            est = sum(len(str(m.get("content", ""))) for m in messages) // 4
            if est <= self.max_context_tokens:
                break
            messages = messages[1:]
        return messages

    # ── Tool call helpers ────────────────────────────────────────────────────

    def _call_tool_sync(self, tc: dict) -> str:
        if self.tracer:
            with self.tracer.span("tool.call", tool=tc["name"]):
                return self.registry.call(
                    tc["name"], tc.get("input", {}), timeout=self.tool_timeout
                )
        return self.registry.call(tc["name"], tc.get("input", {}), timeout=self.tool_timeout)

    def _check_tool_errors(self, tc: dict, result: str, counts: dict[str, int]) -> None:
        if isinstance(result, str) and result.startswith("Error:"):
            counts[tc["name"]] = counts.get(tc["name"], 0) + 1
            if counts[tc["name"]] >= self.max_tool_errors:
                raise RuntimeError(
                    f"Tool '{tc['name']}' failed {self.max_tool_errors} times. Last error: {result}"
                )

    # ── Sync ────────────────────────────────────────────────────────────────

    def run(self, message: str, output_type: type[T] | None = None) -> str | T:
        self.last_usage = {}
        system = self._build_system(message)
        message = self._apply_input_guardrails(message)
        messages = self._build_messages(message)
        tools = self.registry.schemas() if self.registry else None
        _tool_error_counts: dict[str, int] = {}

        for _ in range(self.max_iterations):
            messages = self._trim_messages(messages)
            if self.tracer:
                with self.tracer.span(
                    "llm.complete", model=getattr(self.model, "model", "unknown")
                ):
                    response = self.model.complete(messages, tools=tools, system=system)
            else:
                response = self.model.complete(messages, tools=tools, system=system)

            self.last_usage = {
                k: self.last_usage.get(k, 0) + response.usage.get(k, 0)
                for k in set(self.last_usage) | set(response.usage)
            }

            if response.tool_calls:
                messages.append({"role": "assistant", "content": self._encode_tool_calls(response)})
                if len(response.tool_calls) > 1:
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        results = list(pool.map(self._call_tool_sync, response.tool_calls))
                else:
                    results = [self._call_tool_sync(tc) for tc in response.tool_calls]
                for tc, result in zip(response.tool_calls, results):
                    self._check_tool_errors(tc, result, _tool_error_counts)
                    messages.append(self._tool_result_message(tc, result))
                continue

            if response.content:
                if output_type is not None:
                    schema = _get_schema(output_type)
                    raw = self.model.structured_complete(
                        messages
                        + [
                            {"role": "user", "content": "Return your answer as the specified JSON."}
                        ],
                        schema=schema,
                        system=system,
                    )
                    return _validate(output_type, raw)

                result = self._apply_output_guardrails(response.content)
                if self.memory and hasattr(self.memory, "append"):
                    self.memory.append("user", message)
                    self.memory.append("assistant", result)
                return result

            break

        raise RuntimeError(
            f"Agent did not produce a response after {self.max_iterations} iterations"
        )

    def stream(self, message: str) -> Iterator[str]:
        system = self._build_system(message)
        message = self._apply_input_guardrails(message)
        messages = self._build_messages(message)
        tools = self.registry.schemas() if self.registry else None
        accumulated_text = ""
        _tool_error_counts: dict[str, int] = {}

        for _ in range(self.max_iterations):
            accumulated_text = ""
            tool_calls: list[dict] = []
            messages = self._trim_messages(messages)

            for chunk in self.model.stream(messages, tools=tools, system=system):
                if chunk.text:
                    yield chunk.text
                    accumulated_text += chunk.text
                if chunk.final:
                    tool_calls = chunk.tool_calls

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._encode_tool_calls_raw(accumulated_text, tool_calls),
                    }
                )
                for tc in tool_calls:
                    result = self._call_tool_sync(tc)
                    self._check_tool_errors(tc, result, _tool_error_counts)
                    messages.append(self._tool_result_message(tc, result))
                continue

            if accumulated_text and self.memory:
                self.memory.append("user", message)
                self.memory.append("assistant", accumulated_text)
            return

        raise RuntimeError(
            f"Agent did not produce a response after {self.max_iterations} iterations"
        )

    # ── Async ────────────────────────────────────────────────────────────────

    async def arun(self, message: str, output_type: type[T] | None = None) -> str | T:
        self.last_usage = {}
        system = self._build_system(message)
        message = self._apply_input_guardrails(message)
        messages = self._build_messages(message)
        tools = self.registry.schemas() if self.registry else None
        _tool_error_counts: dict[str, int] = {}

        for _ in range(self.max_iterations):
            messages = self._trim_messages(messages)
            if self.tracer:
                with self.tracer.span(
                    "llm.complete", model=getattr(self.model, "model", "unknown")
                ):
                    response = await self.model.acomplete(messages, tools=tools, system=system)
            else:
                response = await self.model.acomplete(messages, tools=tools, system=system)

            self.last_usage = {
                k: self.last_usage.get(k, 0) + response.usage.get(k, 0)
                for k in set(self.last_usage) | set(response.usage)
            }

            if response.tool_calls:
                messages.append({"role": "assistant", "content": self._encode_tool_calls(response)})
                if self.tracer:
                    results = []
                    for tc in response.tool_calls:
                        with self.tracer.span("tool.call", tool=tc["name"]):
                            results.append(
                                await self.registry.acall(
                                    tc["name"], tc.get("input", {}), timeout=self.tool_timeout
                                )
                            )
                else:
                    results = await asyncio.gather(
                        *[
                            self.registry.acall(
                                tc["name"], tc.get("input", {}), timeout=self.tool_timeout
                            )
                            for tc in response.tool_calls
                        ]
                    )
                for tc, result in zip(response.tool_calls, results):
                    self._check_tool_errors(tc, result, _tool_error_counts)
                    messages.append(self._tool_result_message(tc, result))
                continue

            if response.content:
                if output_type is not None:
                    schema = _get_schema(output_type)
                    raw = self.model.structured_complete(
                        messages
                        + [
                            {"role": "user", "content": "Return your answer as the specified JSON."}
                        ],
                        schema=schema,
                        system=system,
                    )
                    return _validate(output_type, raw)

                result = self._apply_output_guardrails(response.content)
                if self.memory and hasattr(self.memory, "append"):
                    self.memory.append("user", message)
                    self.memory.append("assistant", result)
                return result

            break

        raise RuntimeError(
            f"Agent did not produce a response after {self.max_iterations} iterations"
        )

    async def astream(self, message: str) -> AsyncIterator[str]:
        system = self._build_system(message)
        message = self._apply_input_guardrails(message)
        messages = self._build_messages(message)
        tools = self.registry.schemas() if self.registry else None
        _tool_error_counts: dict[str, int] = {}

        for _ in range(self.max_iterations):
            accumulated_text = ""
            tool_calls: list[dict] = []
            messages = self._trim_messages(messages)

            async for chunk in self.model.astream(messages, tools=tools, system=system):
                if chunk.text:
                    yield chunk.text
                    accumulated_text += chunk.text
                if chunk.final:
                    tool_calls = chunk.tool_calls

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": self._encode_tool_calls_raw(accumulated_text, tool_calls),
                    }
                )
                results = await asyncio.gather(
                    *[
                        self.registry.acall(
                            tc["name"], tc.get("input", {}), timeout=self.tool_timeout
                        )
                        for tc in tool_calls
                    ]
                )
                for tc, result in zip(tool_calls, results):
                    self._check_tool_errors(tc, result, _tool_error_counts)
                    messages.append(self._tool_result_message(tc, result))
                continue

            if accumulated_text and self.memory:
                self.memory.append("user", message)
                self.memory.append("assistant", accumulated_text)
            return

        raise RuntimeError(
            f"Agent did not produce a response after {self.max_iterations} iterations"
        )

    # ── Multi-step planning ──────────────────────────────────────────────────

    def plan_and_run(self, goal: str) -> str:
        """Generate a multi-step plan then execute each step, returning the final answer."""
        # Step 1: generate plan
        plan_prompt = (
            f"Break the following goal into a numbered list of concrete steps (max 5). "
            f"Output ONLY the numbered list, no prose before or after.\n\nGoal: {goal}"
        )
        plan_messages = [{"role": "user", "content": plan_prompt}]
        plan_response = self.model.complete(plan_messages, system=self.instructions)
        plan_text = plan_response.content.strip()

        # Parse steps
        steps = []
        for line in plan_text.splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Strip leading "1." or "1)" or "-"
                step = line.lstrip("0123456789.-) ").strip()
                if step:
                    steps.append(step)

        if not steps:
            # Fallback: run as a single step
            return self.run(goal)

        # Step 2: execute each step with accumulated context
        context_parts: list[str] = [f"Overall goal: {goal}", f"Plan:\n{plan_text}", ""]
        last_result = ""
        for i, step in enumerate(steps, 1):
            context = "\n".join(context_parts)
            step_prompt = f"{context}\nNow execute step {i}: {step}"
            last_result = self.run(step_prompt)
            context_parts.append(f"Step {i} result: {last_result}")

        # Step 3: consolidate
        context = "\n".join(context_parts)
        final_prompt = (
            f"{context}\n\nNow provide a final consolidated answer to the original goal: {goal}"
        )
        return self.run(final_prompt)

    async def aplan_and_run(self, goal: str) -> str:
        """Async version of plan_and_run."""
        plan_prompt = (
            f"Break the following goal into a numbered list of concrete steps (max 5). "
            f"Output ONLY the numbered list, no prose before or after.\n\nGoal: {goal}"
        )
        plan_messages = [{"role": "user", "content": plan_prompt}]
        plan_response = await self.model.acomplete(plan_messages, system=self.instructions)
        plan_text = plan_response.content.strip()

        steps = []
        for line in plan_text.splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                step = line.lstrip("0123456789.-) ").strip()
                if step:
                    steps.append(step)

        if not steps:
            return await self.arun(goal)

        context_parts: list[str] = [f"Overall goal: {goal}", f"Plan:\n{plan_text}", ""]
        last_result = ""
        for i, step in enumerate(steps, 1):
            context = "\n".join(context_parts)
            step_prompt = f"{context}\nNow execute step {i}: {step}"
            last_result = await self.arun(step_prompt)
            context_parts.append(f"Step {i} result: {last_result}")

        context = "\n".join(context_parts)
        final_prompt = (
            f"{context}\n\nNow provide a final consolidated answer to the original goal: {goal}"
        )
        return await self.arun(final_prompt)

    # ── Agent-to-agent messaging ─────────────────────────────────────────────

    def send(self, target: "Agent", message: str) -> str:
        """Send a message to another agent and return its response."""
        return target.run(message)

    async def asend(self, target: "Agent", message: str) -> str:
        """Async send to another agent."""
        return await target.arun(message)

    # ── Internals ────────────────────────────────────────────────────────────

    def _build_system(self, query: str) -> str:
        parts = []
        if self.instructions:
            parts.append(self.instructions)
        if self.knowledge:
            chunks = self.knowledge.search(query)
            if chunks:
                parts.append("Relevant context:\n" + "\n---\n".join(chunks))
        if self.memory and hasattr(self.memory, "all_facts"):
            if (
                hasattr(self.memory, "semantic_recall")
                and getattr(self.memory, "embed_fn", None) is not None
            ):
                facts = dict(self.memory.semantic_recall(query, top_k=5))
            else:
                facts = self.memory.all_facts()
            if facts:
                lines = [f"- {k}: {v}" for k, v in facts.items()]
                parts.append("Known facts:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _build_messages(self, message: str) -> list[dict]:
        history = self.memory.messages() if self.memory and hasattr(self.memory, "messages") else []
        return list(history) + [{"role": "user", "content": message}]

    def _encode_tool_calls(self, response) -> list[dict]:
        blocks = []
        if response.content:
            blocks.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc.get("input", {}),
                }
            )
        return blocks

    def _encode_tool_calls_raw(self, text: str, tool_calls: list[dict]) -> list[dict]:
        blocks = []
        if text:
            blocks.append({"type": "text", "text": text})
        for tc in tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc.get("input", {}),
                }
            )
        return blocks

    def _tool_result_message(self, tc: dict, result: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tc["id"], "content": result}],
        }


def _get_schema(output_type: type) -> dict:
    if hasattr(output_type, "model_json_schema"):
        return output_type.model_json_schema()
    raise TypeError(f"{output_type} is not a Pydantic model. Pass a Pydantic BaseModel subclass.")


def _validate(output_type: type[T], raw: dict) -> T:
    if hasattr(output_type, "model_validate"):
        return output_type.model_validate(raw)
    raise TypeError(f"{output_type} does not support model_validate.")
