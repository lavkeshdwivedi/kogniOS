from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TypeVar

import json
from collections.abc import Iterable
from datetime import datetime, timezone

from .checkpoint import (
    DONE,
    INTERRUPTED,
    RUNNING,
    ApprovalRequired,
    Checkpoint,
    CheckpointStore,
    save_record,
)
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
        checkpointer: CheckpointStore | None = None,
        require_approval: Iterable[str] | None = None,
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
        self.checkpointer = checkpointer
        self.require_approval = frozenset(require_approval or ())
        if self.require_approval and checkpointer is None:
            raise ValueError("require_approval needs a checkpointer so a paused run can be resumed")

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

    def run(
        self,
        message: str,
        output_type: type[T] | None = None,
        run_id: str | None = None,
    ) -> str | T:
        """Run the agent to an answer.

        With a checkpointer and a run_id, every step is saved. Calling run again with the same
        run_id continues where the last step stopped, and tools named in require_approval
        pause the run with ApprovalRequired until agent.resume() supplies a decision.
        """
        cp = self._load_run(run_id)
        if cp is not None:
            return self._continue(cp, output_type, None)
        if self.require_approval and not run_id:
            raise ValueError("require_approval needs a run_id so the run can be resumed")
        self.last_usage = {}
        system = self._build_system(message)
        message = self._apply_input_guardrails(message)
        messages = self._build_messages(message)
        state = Checkpoint(
            run_id=run_id or "", step=0, status=RUNNING, message=message, messages=messages
        )
        self._checkpoint(state, RUNNING)
        return self._run_loop(state, system, output_type, None)

    def resume(
        self,
        run_id: str,
        decisions: dict[str, bool | dict] | None = None,
        output_type: type[T] | None = None,
    ) -> str | T:
        """Continue a saved run. decisions maps a tool call id to True (approve), False
        (reject) or {"approve": bool, "input": {...}, "reason": str} to edit the arguments."""
        cp = self._load_run(run_id)
        if cp is None:
            raise KeyError(f"No saved run '{run_id}'")
        return self._continue(cp, output_type, decisions)

    def _load_run(self, run_id: str | None) -> Checkpoint | None:
        if run_id is None:
            return None
        if self.checkpointer is None:
            raise ValueError("run_id needs an Agent(checkpointer=...)")
        return self.checkpointer.load(run_id)

    def _continue(
        self, cp: Checkpoint, output_type: type[T] | None, decisions: dict[str, bool | dict] | None
    ) -> str | T:
        if cp.status == DONE and cp.result is not None:
            if output_type is not None:
                return _validate(output_type, json.loads(cp.result))
            return cp.result
        self.last_usage = dict(cp.usage)
        return self._run_loop(cp, self._build_system(cp.message), output_type, decisions)

    def _checkpoint(self, state: Checkpoint, status: str) -> None:
        if self.checkpointer is None or not state.run_id:
            return
        state.status = status
        state.step += 1
        state.usage = dict(self.last_usage)
        state.created_at = datetime.now(timezone.utc).isoformat()
        self.checkpointer.save(state)

    async def _acheckpoint(self, state: Checkpoint, status: str) -> None:
        # the SQLite write runs in a worker thread so the event loop keeps serving other tasks
        if self.checkpointer is None or not state.run_id:
            return
        await asyncio.to_thread(self._checkpoint, state, status)

    def _run_or_resume(self, message: str, run_id: str | None, decisions) -> str:
        """run() for a fresh or crashed run, resume() when it is paused and decisions came in."""
        cp = self._load_run(run_id)
        if cp is not None and cp.status == INTERRUPTED and decisions:
            return self.resume(run_id, decisions)  # type: ignore[arg-type,return-value]
        return self.run(message, run_id=run_id)  # type: ignore[return-value]

    async def _arun_or_resume(self, message: str, run_id: str | None, decisions) -> str:
        cp = await asyncio.to_thread(self._load_run, run_id)
        if cp is not None and cp.status == INTERRUPTED and decisions:
            return await self.aresume(run_id, decisions)  # type: ignore[arg-type,return-value]
        return await self.arun(message, run_id=run_id)  # type: ignore[return-value]

    def _run_loop(
        self,
        state: Checkpoint,
        system: str,
        output_type: type[T] | None,
        decisions: dict[str, bool | dict] | None,
    ) -> str | T:
        tools = self.registry.schemas() if self.registry else None
        messages = state.messages
        held = state.pending_calls or None  # tool calls saved before the run was paused

        while state.iterations_used < self.max_iterations:
            if held is not None:
                calls, held = held, None
            else:
                messages = self._trim_messages(messages)
                state.messages = messages
                if self.tracer:
                    with self.tracer.span(
                        "llm.complete", model=getattr(self.model, "model", "unknown")
                    ):
                        response = self.model.complete(messages, tools=tools, system=system)
                else:
                    response = self.model.complete(messages, tools=tools, system=system)
                state.iterations_used += 1

                self.last_usage = {
                    k: self.last_usage.get(k, 0) + response.usage.get(k, 0)
                    for k in set(self.last_usage) | set(response.usage)
                }

                if not response.tool_calls:
                    if not response.content:
                        break
                    if output_type is not None:
                        schema = _get_schema(output_type)
                        raw = self.model.structured_complete(
                            messages
                            + [
                                {
                                    "role": "user",
                                    "content": "Return your answer as the specified JSON.",
                                }
                            ],
                            schema=schema,
                            system=system,
                        )
                        state.result = json.dumps(raw)
                        self._checkpoint(state, DONE)
                        return _validate(output_type, raw)

                    result = self._apply_output_guardrails(response.content)
                    if self.memory and hasattr(self.memory, "append"):
                        self.memory.append("user", state.message)
                        self.memory.append("assistant", result)
                    state.result = result
                    self._checkpoint(state, DONE)
                    return result

                messages.append({"role": "assistant", "content": self._encode_tool_calls(response)})
                calls = response.tool_calls

            results = self._execute_calls(state, calls, decisions)
            decisions = None
            for tc, result in zip(calls, results):
                self._check_tool_errors(tc, result, state.tool_error_counts)
                messages.append(self._tool_result_message(tc, result))
            state.pending_calls, state.awaiting = [], []
            self._checkpoint(state, RUNNING)

        raise RuntimeError(
            f"Agent did not produce a response after {self.max_iterations} iterations"
        )

    def _resolve_calls(
        self,
        state: Checkpoint,
        calls: list[dict],
        decisions: dict[str, bool | dict] | None,
    ) -> tuple[dict[str, str], list[dict]]:
        """Apply reviewer decisions to one model turn. Returns the results already known
        (rejections) and the calls that still need to run, or raises ApprovalRequired."""
        decided = decisions or {}
        gated = [tc for tc in calls if tc["name"] in self.require_approval]
        awaiting = [tc["id"] for tc in gated if tc["id"] not in decided]
        if awaiting:
            # nothing from this turn has run yet, so resuming cannot repeat a side effect
            state.pending_calls = calls
            state.awaiting = awaiting
            self._checkpoint(state, INTERRUPTED)
            raise ApprovalRequired(state.run_id, [tc for tc in calls if tc["id"] in awaiting])

        results: dict[str, str] = {}
        to_run: list[dict] = []
        for tc in calls:
            d = decided.get(tc["id"]) if tc["name"] in self.require_approval else None
            if d is None:
                to_run.append(tc)
                continue
            approve = d if isinstance(d, bool) else bool(d.get("approve", True))
            if not approve:
                reason = "" if isinstance(d, bool) else d.get("reason", "")
                results[tc["id"]] = f"Tool call '{tc['name']}' was rejected by the reviewer." + (
                    f" Reason: {reason}" if reason else ""
                )
                continue
            edited = None if isinstance(d, bool) else d.get("input")
            if edited is not None:
                tc = {**tc, "input": edited}
                self._record_edit(state, tc)
            to_run.append(tc)
        return results, to_run

    def _execute_calls(
        self,
        state: Checkpoint,
        calls: list[dict],
        decisions: dict[str, bool | dict] | None,
    ) -> list[str]:
        """Run one model turn's tool calls, holding back any that need a reviewer first."""
        results, to_run = self._resolve_calls(state, calls, decisions)
        if len(to_run) > 1:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                outputs = list(pool.map(self._call_tool_sync, to_run))
        else:
            outputs = [self._call_tool_sync(tc) for tc in to_run]
        results.update({tc["id"]: out for tc, out in zip(to_run, outputs)})
        return [results[tc["id"]] for tc in calls]

    def _record_edit(self, state: Checkpoint, tc: dict) -> None:
        # keep the saved assistant turn honest: the model should see what actually ran
        for block in state.messages[-1].get("content", []):
            if isinstance(block, dict) and block.get("id") == tc["id"]:
                block["input"] = tc["input"]

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

    async def arun(
        self,
        message: str,
        output_type: type[T] | None = None,
        run_id: str | None = None,
    ) -> str | T:
        """Async run with the same checkpoint, resume and approval semantics as run()."""
        cp = await asyncio.to_thread(self._load_run, run_id)
        if cp is not None:
            return await self._acontinue(cp, output_type, None)
        if self.require_approval and not run_id:
            raise ValueError("require_approval needs a run_id so the run can be resumed")
        self.last_usage = {}
        system = self._build_system(message)
        message = self._apply_input_guardrails(message)
        messages = self._build_messages(message)
        state = Checkpoint(
            run_id=run_id or "", step=0, status=RUNNING, message=message, messages=messages
        )
        await self._acheckpoint(state, RUNNING)
        return await self._arun_loop(state, system, output_type, None)

    async def aresume(
        self,
        run_id: str,
        decisions: dict[str, bool | dict] | None = None,
        output_type: type[T] | None = None,
    ) -> str | T:
        """Async resume(). Same decision format."""
        cp = await asyncio.to_thread(self._load_run, run_id)
        if cp is None:
            raise KeyError(f"No saved run '{run_id}'")
        return await self._acontinue(cp, output_type, decisions)

    async def _acontinue(
        self, cp: Checkpoint, output_type: type[T] | None, decisions: dict[str, bool | dict] | None
    ) -> str | T:
        if cp.status == DONE and cp.result is not None:
            if output_type is not None:
                return _validate(output_type, json.loads(cp.result))
            return cp.result
        self.last_usage = dict(cp.usage)
        return await self._arun_loop(cp, self._build_system(cp.message), output_type, decisions)

    async def _arun_loop(
        self,
        state: Checkpoint,
        system: str,
        output_type: type[T] | None,
        decisions: dict[str, bool | dict] | None,
    ) -> str | T:
        tools = self.registry.schemas() if self.registry else None
        messages = state.messages
        held = state.pending_calls or None

        while state.iterations_used < self.max_iterations:
            if held is not None:
                calls, held = held, None
            else:
                messages = self._trim_messages(messages)
                state.messages = messages
                if self.tracer:
                    with self.tracer.span(
                        "llm.complete", model=getattr(self.model, "model", "unknown")
                    ):
                        response = await self.model.acomplete(messages, tools=tools, system=system)
                else:
                    response = await self.model.acomplete(messages, tools=tools, system=system)
                state.iterations_used += 1

                self.last_usage = {
                    k: self.last_usage.get(k, 0) + response.usage.get(k, 0)
                    for k in set(self.last_usage) | set(response.usage)
                }

                if not response.tool_calls:
                    if not response.content:
                        break
                    if output_type is not None:
                        schema = _get_schema(output_type)
                        raw = self.model.structured_complete(
                            messages
                            + [
                                {
                                    "role": "user",
                                    "content": "Return your answer as the specified JSON.",
                                }
                            ],
                            schema=schema,
                            system=system,
                        )
                        state.result = json.dumps(raw)
                        await self._acheckpoint(state, DONE)
                        return _validate(output_type, raw)

                    result = self._apply_output_guardrails(response.content)
                    if self.memory and hasattr(self.memory, "append"):
                        self.memory.append("user", state.message)
                        self.memory.append("assistant", result)
                    state.result = result
                    await self._acheckpoint(state, DONE)
                    return result

                messages.append({"role": "assistant", "content": self._encode_tool_calls(response)})
                calls = response.tool_calls

            results = await self._aexecute_calls(state, calls, decisions)
            decisions = None
            for tc, result in zip(calls, results):
                self._check_tool_errors(tc, result, state.tool_error_counts)
                messages.append(self._tool_result_message(tc, result))
            state.pending_calls, state.awaiting = [], []
            await self._acheckpoint(state, RUNNING)

        raise RuntimeError(
            f"Agent did not produce a response after {self.max_iterations} iterations"
        )

    async def _aexecute_calls(
        self,
        state: Checkpoint,
        calls: list[dict],
        decisions: dict[str, bool | dict] | None,
    ) -> list[str]:
        results, to_run = await asyncio.to_thread(self._resolve_calls, state, calls, decisions)
        if self.tracer:
            outputs = []
            for tc in to_run:
                with self.tracer.span("tool.call", tool=tc["name"]):
                    outputs.append(
                        await self.registry.acall(
                            tc["name"], tc.get("input", {}), timeout=self.tool_timeout
                        )
                    )
        else:
            outputs = await asyncio.gather(
                *[
                    self.registry.acall(tc["name"], tc.get("input", {}), timeout=self.tool_timeout)
                    for tc in to_run
                ]
            )
        results.update({tc["id"]: out for tc, out in zip(to_run, outputs)})
        return [results[tc["id"]] for tc in calls]

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

    async def astream_events(
        self,
        message: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> AsyncIterator[dict]:
        """Stream a run as step events, with checkpoints when run_id is given.

        Events are dicts with a "type": text, tool_call, tool_result, approval_required and
        done. A gated tool call ends the stream with approval_required (the run is saved);
        call again with the same run_id and decisions to continue. Plain astream() is unchanged
        and still yields text only. Output guardrails cannot rewrite text already streamed, so
        they are not applied here.
        """
        cp = await asyncio.to_thread(self._load_run, run_id)
        if cp is not None and cp.status == DONE and cp.result is not None:
            yield {"type": "done", "run_id": run_id, "result": cp.result}
            return
        if cp is None and self.require_approval and not run_id:
            raise ValueError("require_approval needs a run_id so the run can be resumed")
        if cp is None:
            self.last_usage = {}
            system = self._build_system(message)
            message = self._apply_input_guardrails(message)
            state = Checkpoint(
                run_id=run_id or "",
                step=0,
                status=RUNNING,
                message=message,
                messages=self._build_messages(message),
            )
            await self._acheckpoint(state, RUNNING)
        else:
            state = cp
            self.last_usage = dict(cp.usage)
            system = self._build_system(cp.message)

        tools = self.registry.schemas() if self.registry else None
        messages = state.messages
        held = state.pending_calls or None

        while state.iterations_used < self.max_iterations:
            if held is not None:
                calls, held = held, None
            else:
                messages = self._trim_messages(messages)
                state.messages = messages
                text = ""
                calls = []
                async for chunk in self.model.astream(messages, tools=tools, system=system):
                    if chunk.text:
                        text += chunk.text
                        yield {"type": "text", "text": chunk.text}
                    if chunk.final:
                        calls = chunk.tool_calls
                state.iterations_used += 1

                if not calls:
                    if self.memory and text:
                        self.memory.append("user", state.message)
                        self.memory.append("assistant", text)
                    state.result = text
                    await self._acheckpoint(state, DONE)
                    yield {"type": "done", "run_id": run_id, "result": text}
                    return
                messages.append(
                    {"role": "assistant", "content": self._encode_tool_calls_raw(text, calls)}
                )

            try:
                results = await self._aexecute_calls(state, calls, decisions)
            except ApprovalRequired as e:
                yield {"type": "approval_required", "run_id": e.run_id, "pending": e.pending}
                return
            decisions = None
            for tc, result in zip(calls, results):
                yield {
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc.get("input", {}),
                }
                yield {"type": "tool_result", "id": tc["id"], "name": tc["name"], "result": result}
                self._check_tool_errors(tc, result, state.tool_error_counts)
                messages.append(self._tool_result_message(tc, result))
            state.pending_calls, state.awaiting = [], []
            await self._acheckpoint(state, RUNNING)

        raise RuntimeError(
            f"Agent did not produce a response after {self.max_iterations} iterations"
        )

    # ── Multi-step planning ──────────────────────────────────────────────────

    @staticmethod
    def _plan_prompt(goal: str) -> str:
        return (
            f"Break the following goal into a numbered list of concrete steps (max 5). "
            f"Output ONLY the numbered list, no prose before or after.\n\nGoal: {goal}"
        )

    @staticmethod
    def _parse_steps(plan_text: str) -> list[str]:
        steps = []
        for line in plan_text.splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # strip a leading "1." or "1)" or "-"
                step = line.lstrip("0123456789.-) ").strip()
                if step:
                    steps.append(step)
        return steps

    def _finished_plan(self, run_id: str | None) -> Checkpoint | None:
        """The saved top-level record when this plan_and_run already finished."""
        top = self._load_run(run_id)
        if top is None:
            return None
        if top.kind != "plan":
            raise ValueError(f"run_id '{run_id}' belongs to a plain run, not a plan_and_run")
        return top

    def plan_and_run(
        self,
        goal: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> str:
        """Generate a multi-step plan then execute each step, returning the final answer.

        With a checkpointer and a run_id the plan and every step are saved under
        ``<run_id>/plan``, ``<run_id>/step-N`` and ``<run_id>/final``. Calling again with the
        same run_id skips finished work and continues the step that stopped. A step that
        paused for approval raises ApprovalRequired, and the next call passes ``decisions``.
        """

        def sub(name: str) -> str | None:
            return f"{run_id}/{name}" if run_id else None

        top = self._finished_plan(run_id)
        if top is not None and top.result is not None:
            return top.result
        saved_plan = self._load_run(sub("plan"))
        if saved_plan is not None and saved_plan.plan is not None:
            plan_text = saved_plan.plan
        else:
            plan_messages = [{"role": "user", "content": self._plan_prompt(goal)}]
            plan_text = self.model.complete(plan_messages, system=self.instructions).content.strip()
            if run_id and self.checkpointer is not None:
                save_record(self.checkpointer, f"{run_id}/plan", "plan", goal, plan=plan_text)
        steps = self._parse_steps(plan_text)

        if not steps:  # fallback: run as a single step
            answer = self._run_or_resume(goal, sub("single"), decisions)
        else:
            context_parts: list[str] = [f"Overall goal: {goal}", f"Plan:\n{plan_text}", ""]
            for i, step in enumerate(steps, 1):
                context = "\n".join(context_parts)
                step_prompt = f"{context}\nNow execute step {i}: {step}"
                last = self._run_or_resume(step_prompt, sub(f"step-{i}"), decisions)
                context_parts.append(f"Step {i} result: {last}")
            context = "\n".join(context_parts)
            final_prompt = (
                f"{context}\n\nNow provide a final consolidated answer to the original goal: {goal}"
            )
            answer = self._run_or_resume(final_prompt, sub("final"), decisions)
        if run_id and self.checkpointer is not None:
            save_record(self.checkpointer, run_id, "plan", goal, result=answer, plan=plan_text)
        return answer

    async def aplan_and_run(
        self,
        goal: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> str:
        """Async plan_and_run() with the same checkpoint, resume and approval behavior."""

        def sub(name: str) -> str | None:
            return f"{run_id}/{name}" if run_id else None

        top = await asyncio.to_thread(self._finished_plan, run_id)
        if top is not None and top.result is not None:
            return top.result
        saved_plan = await asyncio.to_thread(self._load_run, sub("plan"))
        if saved_plan is not None and saved_plan.plan is not None:
            plan_text = saved_plan.plan
        else:
            plan_messages = [{"role": "user", "content": self._plan_prompt(goal)}]
            plan_response = await self.model.acomplete(plan_messages, system=self.instructions)
            plan_text = plan_response.content.strip()
            if run_id and self.checkpointer is not None:
                await asyncio.to_thread(
                    save_record, self.checkpointer, f"{run_id}/plan", "plan", goal, None, plan_text
                )
        steps = self._parse_steps(plan_text)

        if not steps:
            answer = await self._arun_or_resume(goal, sub("single"), decisions)
        else:
            context_parts: list[str] = [f"Overall goal: {goal}", f"Plan:\n{plan_text}", ""]
            for i, step in enumerate(steps, 1):
                context = "\n".join(context_parts)
                step_prompt = f"{context}\nNow execute step {i}: {step}"
                last = await self._arun_or_resume(step_prompt, sub(f"step-{i}"), decisions)
                context_parts.append(f"Step {i} result: {last}")
            context = "\n".join(context_parts)
            final_prompt = (
                f"{context}\n\nNow provide a final consolidated answer to the original goal: {goal}"
            )
            answer = await self._arun_or_resume(final_prompt, sub("final"), decisions)
        if run_id and self.checkpointer is not None:
            await asyncio.to_thread(
                save_record, self.checkpointer, run_id, "plan", goal, answer, plan_text
            )
        return answer

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
