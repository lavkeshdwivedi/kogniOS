from __future__ import annotations

import asyncio
import concurrent.futures
import json
import re

from .checkpoint import CheckpointStore, load_record, save_record
from .models.base import BaseModel


class Team:
    """Routes work to agents.

    Pass ``checkpointer`` and a ``run_id`` to run, pipeline or broadcast to make the flow
    resumable. The team saves its routing decision, each member saves its own run under
    ``<run_id>/<agent>`` (so members need their own checkpointer too), and calling again with
    the same run_id skips finished work. A member paused for approval raises
    ApprovalRequired, and the next call passes ``decisions``.
    """

    def __init__(
        self,
        router_model: BaseModel,
        agents: dict,
        instructions: str = "",
        checkpointer: CheckpointStore | None = None,
    ):
        self.router_model = router_model
        self.agents = agents
        self.instructions = instructions
        self.checkpointer = checkpointer

    # ── helpers ──────────────────────────────────────────────────────────────

    def _need_store(self, run_id: str | None) -> None:
        if run_id and self.checkpointer is None:
            raise ValueError("run_id needs a Team(checkpointer=...)")

    def _exec(self, agent, message: str, sub_id: str | None, decisions) -> str:
        if sub_id is None:  # no run_id: the plain, unsaved path
            return agent.run(message)
        return agent._run_or_resume(message, sub_id, decisions)

    async def _aexec(self, agent, message: str, sub_id: str | None, decisions) -> str:
        if sub_id is None:
            return await agent.arun(message)
        return await agent._arun_or_resume(message, sub_id, decisions)

    def _router_system(self) -> str:
        agent_list = "\n".join(
            f'- "{name}": {agent.description or "(no description)"}'
            for name, agent in self.agents.items()
        )
        return (
            (self.instructions + "\n\n" if self.instructions else "")
            + f"You are a router. Available agents:\n{agent_list}\n\n"
            "Reply with a JSON object: "
            '{"agent": "<name>", "message": "<task for the agent>"}'
            "\nChoose the most appropriate agent for the user's request."
        )

    def _saved_route(self, run_id: str | None) -> tuple[str, str] | None:
        if not run_id or self.checkpointer is None:
            return None
        rec = load_record(self.checkpointer, f"{run_id}/route")
        if rec is None or rec.result is None:
            return None
        data = json.loads(rec.result)
        return data["agent"], data["message"]

    def _resolve(self, name: str, forwarded: str) -> tuple[str, str]:
        if name not in self.agents:
            name = next(iter(self.agents))
        return name, forwarded

    def _save_route(self, run_id: str | None, message: str, name: str, forwarded: str) -> None:
        if run_id and self.checkpointer is not None:
            payload = json.dumps({"agent": name, "message": forwarded})
            save_record(self.checkpointer, f"{run_id}/route", "route", message, result=payload)

    # ── routed run ───────────────────────────────────────────────────────────

    def run(
        self,
        message: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> str:
        self._need_store(run_id)
        route = self._saved_route(run_id)
        if route is None:
            response = self.router_model.complete(
                [{"role": "user", "content": message}],
                system=self._router_system(),
            )
            route = self._resolve(*self._parse_routing(response.content, message))
            self._save_route(run_id, message, *route)
        name, forwarded = route
        agent = self.agents[name]
        return self._exec(agent, forwarded, f"{run_id}/{name}" if run_id else None, decisions)

    def resume(self, run_id: str, decisions: dict[str, bool | dict] | None = None) -> str:
        """Continue a routed run with reviewer decisions, without routing again."""
        self._need_store(run_id)
        rec = load_record(self.checkpointer, f"{run_id}/route")  # type: ignore[arg-type]
        if rec is None:
            raise KeyError(f"No saved team run '{run_id}'")
        return self.run(rec.message, run_id=run_id, decisions=decisions)

    async def arun(
        self,
        message: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> str:
        self._need_store(run_id)
        route = await asyncio.to_thread(self._saved_route, run_id)
        if route is None:
            response = await self.router_model.acomplete(
                [{"role": "user", "content": message}],
                system=self._router_system(),
            )
            route = self._resolve(*self._parse_routing(response.content, message))
            await asyncio.to_thread(self._save_route, run_id, message, *route)
        name, forwarded = route
        agent = self.agents[name]
        return await self._aexec(
            agent, forwarded, f"{run_id}/{name}" if run_id else None, decisions
        )

    async def aresume(self, run_id: str, decisions: dict[str, bool | dict] | None = None) -> str:
        self._need_store(run_id)
        rec = await asyncio.to_thread(load_record, self.checkpointer, f"{run_id}/route")  # type: ignore[arg-type]
        if rec is None:
            raise KeyError(f"No saved team run '{run_id}'")
        return await self.arun(rec.message, run_id=run_id, decisions=decisions)

    # ── pipeline and fan-out ─────────────────────────────────────────────────

    def pipeline(
        self,
        message: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> str:
        """Sequential chain: output of each agent becomes input to the next."""
        self._need_store(run_id)
        result = message
        for i, (name, agent) in enumerate(self.agents.items()):
            sub_id = f"{run_id}/stage-{i}-{name}" if run_id else None
            result = self._exec(agent, result, sub_id, decisions)
        return result

    async def apipeline(
        self,
        message: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> str:
        """Async sequential chain."""
        self._need_store(run_id)
        result = message
        for i, (name, agent) in enumerate(self.agents.items()):
            sub_id = f"{run_id}/stage-{i}-{name}" if run_id else None
            result = await self._aexec(agent, result, sub_id, decisions)
        return result

    def broadcast(
        self,
        message: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> dict[str, str]:
        """Fan-out to all agents in parallel, return {name: result}.

        If one agent fails or pauses, the others still finish and are saved, so calling again
        with the same run_id only redoes the one that stopped.
        """
        self._need_store(run_id)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {
                name: pool.submit(
                    self._exec, agent, message, f"{run_id}/{name}" if run_id else None, decisions
                )
                for name, agent in self.agents.items()
            }
            return {name: f.result() for name, f in futures.items()}

    async def abroadcast(
        self,
        message: str,
        run_id: str | None = None,
        decisions: dict[str, bool | dict] | None = None,
    ) -> dict[str, str]:
        """Async fan-out to all agents in parallel, return {name: result}."""
        self._need_store(run_id)
        outcomes = await asyncio.gather(
            *[
                self._aexec(agent, message, f"{run_id}/{name}" if run_id else None, decisions)
                for name, agent in self.agents.items()
            ],
            return_exceptions=bool(run_id),  # let the others save before surfacing a failure
        )
        for out in outcomes:
            if isinstance(out, BaseException):
                raise out
        return dict(zip(self.agents.keys(), outcomes))

    def _parse_routing(self, text: str, fallback_message: str) -> tuple[str, str]:
        try:
            m = re.search(r"\{[^}]+\}", text, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return data["agent"], data.get("message", fallback_message)
        except Exception:
            pass
        return next(iter(self.agents)), fallback_message
