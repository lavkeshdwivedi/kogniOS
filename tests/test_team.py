import pytest
from unittest.mock import MagicMock
from kognios.team import Team
from kognios.models.base import ModelResponse


def make_agent(response: str, name: str = "", description: str = ""):
    agent = MagicMock()
    agent.run.return_value = response
    agent.arun = MagicMock(return_value=response)
    agent.description = description
    agent.name = name
    return agent


def make_router(routing_json: str):
    model = MagicMock()
    model.complete.return_value = ModelResponse(content=routing_json, tool_calls=[])
    return model


def test_team_run_routes_to_correct_agent():
    a1 = make_agent("answer from agent1", description="handles math")
    a2 = make_agent("answer from agent2", description="handles writing")
    router = make_router('{"agent": "math", "message": "what is 2+2?"}')
    team = Team(router_model=router, agents={"math": a1, "writing": a2})
    result = team.run("what is 2+2?")
    assert result == "answer from agent1"
    a1.run.assert_called_once_with("what is 2+2?")
    a2.run.assert_not_called()


def test_team_run_falls_back_on_bad_json():
    a1 = make_agent("fallback answer")
    router = make_router("not json at all")
    team = Team(router_model=router, agents={"only": a1})
    result = team.run("hello")
    assert result == "fallback answer"


def test_team_pipeline_chains_outputs():
    a1 = make_agent("step1_result")
    a2 = make_agent("step2_result")
    a2.run.side_effect = lambda msg: f"processed:{msg}"
    team = Team(router_model=MagicMock(), agents={"a": a1, "b": a2})
    result = team.pipeline("input")
    a1.run.assert_called_once_with("input")
    a2.run.assert_called_once_with("step1_result")
    assert result == "processed:step1_result"


def test_team_broadcast_returns_all_results():
    a1 = make_agent("r1")
    a2 = make_agent("r2")
    team = Team(router_model=MagicMock(), agents={"alpha": a1, "beta": a2})
    results = team.broadcast("same message")
    assert results == {"alpha": "r1", "beta": "r2"}
    a1.run.assert_called_once_with("same message")
    a2.run.assert_called_once_with("same message")


@pytest.mark.asyncio
async def test_team_apipeline_chains_outputs():
    async def _arun_a1(msg):
        return "async_step1"

    async def _arun_a2(msg):
        return f"async_processed:{msg}"

    a1 = MagicMock()
    a1.description = ""
    a1.arun = _arun_a1
    a2 = MagicMock()
    a2.description = ""
    a2.arun = _arun_a2

    team = Team(router_model=MagicMock(), agents={"a": a1, "b": a2})
    result = await team.apipeline("start")
    assert result == "async_processed:async_step1"


@pytest.mark.asyncio
async def test_team_abroadcast_returns_all_results():
    async def _arun(msg):
        return f"result_for_{msg}"

    a1 = MagicMock()
    a1.description = ""
    a1.arun = _arun
    a2 = MagicMock()
    a2.description = ""
    a2.arun = _arun

    team = Team(router_model=MagicMock(), agents={"x": a1, "y": a2})
    results = await team.abroadcast("hello")
    assert results == {"x": "result_for_hello", "y": "result_for_hello"}
