from __future__ import annotations

import json
import re

from .models.base import BaseModel


class Team:
    def __init__(
        self,
        router_model: BaseModel,
        agents: dict,
        instructions: str = "",
    ):
        self.router_model = router_model
        self.agents = agents
        self.instructions = instructions

    def run(self, message: str) -> str:
        agent_list = "\n".join(
            f'- "{name}": {agent.description or "(no description)"}'
            for name, agent in self.agents.items()
        )
        system = (
            (self.instructions + "\n\n" if self.instructions else "")
            + f"You are a router. Available agents:\n{agent_list}\n\n"
            "Reply with a JSON object: "
            '{"agent": "<name>", "message": "<task for the agent>"}'
            "\nChoose the most appropriate agent for the user's request."
        )
        response = self.router_model.complete(
            [{"role": "user", "content": message}],
            system=system,
        )
        name, forwarded = self._parse_routing(response.content, message)
        agent = self.agents.get(name, next(iter(self.agents.values())))
        return agent.run(forwarded)

    def _parse_routing(self, text: str, fallback_message: str) -> tuple[str, str]:
        try:
            m = re.search(r"\{[^}]+\}", text, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return data["agent"], data.get("message", fallback_message)
        except Exception:
            pass
        return next(iter(self.agents)), fallback_message
