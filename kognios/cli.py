from __future__ import annotations

import json
import uuid

import click

from .agent import Agent
from .memory.short_term import ShortTermMemory
from .models.anthropic import AnthropicModel
from .models.openai import OpenAIModel

_PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "groq": "llama-4-scout",
    "gemini": "gemini-2.5-flash",
    "mistral": "mistral-large-latest",
    "cohere": "command-a-plus-05-2026",
    "ollama": "llama3.3",
    "bedrock": "anthropic.claude-sonnet-5",
    "xai": "grok-4",
}

_ALL_PROVIDERS = list(_PROVIDER_DEFAULTS.keys())


def _make_model(provider: str, model: str):
    if provider == "anthropic":
        return AnthropicModel(model=model)
    if provider == "openai":
        return OpenAIModel(model=model)
    if provider == "groq":
        from .models.groq import GroqModel

        return GroqModel(model=model)
    if provider == "gemini":
        from .models.gemini import GeminiModel

        return GeminiModel(model=model)
    if provider == "mistral":
        from .models.mistral import MistralModel

        return MistralModel(model=model)
    if provider == "cohere":
        from .models.cohere import CohereModel

        return CohereModel(model=model)
    if provider == "ollama":
        from .models.ollama import OllamaModel

        return OllamaModel(model=model)
    if provider == "bedrock":
        from .models.bedrock import BedrockModel

        return BedrockModel(model=model)
    if provider == "xai":
        from .models.xai import XAIModel

        return XAIModel(model=model)
    raise click.BadParameter(f"Unknown provider: {provider}")


@click.group()
@click.version_option(package_name="kognios")
def cli():
    """Kognios: cognitive agent framework."""


@cli.command()
@click.option(
    "--provider",
    "-p",
    default="anthropic",
    type=click.Choice(_ALL_PROVIDERS),
    help="LLM provider.",
)
@click.option("--model", "-m", default=None, help="Model name (defaults per provider).")
@click.option(
    "--instructions",
    "-i",
    default="You are a helpful assistant.",
    help="System instructions for the agent.",
)
@click.option("--stream/--no-stream", default=True, help="Stream response tokens.")
def chat(provider: str, model: str | None, instructions: str, stream: bool):
    """Interactive chat with a Kognios agent."""
    chosen_model = model or _PROVIDER_DEFAULTS[provider]
    llm = _make_model(provider, chosen_model)
    memory = ShortTermMemory(max_turns=20)
    agent = Agent(model=llm, memory=memory, instructions=instructions)

    click.echo(click.style(f"Kognios chat | {provider}/{chosen_model}", fg="cyan"))
    click.echo(click.style("Type 'exit' or Ctrl-C to quit.\n", fg="bright_black"))

    while True:
        try:
            user_input = click.prompt(click.style("You", fg="green"))
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye.")
            break

        if user_input.strip().lower() in ("exit", "quit", "bye", "q"):
            click.echo("Bye.")
            break

        click.echo(click.style("Agent: ", fg="yellow"), nl=False)
        try:
            if stream:
                for chunk in agent.stream(user_input):
                    click.echo(chunk, nl=False)
                click.echo()
            else:
                response = agent.run(user_input)
                click.echo(response)
        except Exception as exc:
            click.echo(click.style(f"\nError: {exc}", fg="red"))
        click.echo()


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.argument("question")
@click.option(
    "--provider",
    "-p",
    default="anthropic",
    type=click.Choice(_ALL_PROVIDERS),
)
@click.option("--model", "-m", default=None)
def ask(file: str, question: str, provider: str, model: str | None):
    """Load a document into a knowledge base and answer a question about it."""
    from .knowledge.sqlite_fts import SQLiteKnowledge

    chosen_model = model or _PROVIDER_DEFAULTS[provider]
    llm = _make_model(provider, chosen_model)
    kb = SQLiteKnowledge(db_path=":memory:")
    kb.load(file)
    agent = Agent(model=llm, knowledge=kb, instructions="Answer only from the provided context.")
    click.echo(agent.run(question))


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--db", default="kognios_kb.db", show_default=True, help="SQLite database path.")
@click.option(
    "--backend",
    default="fts",
    type=click.Choice(["fts", "vector"]),
    help="Storage backend: fts (BM25) or vector (cosine similarity).",
)
def ingest(file: str, db: str, backend: str):
    """Ingest a document into a knowledge base.

    Supports plain text, PDF, HTML, CSV, JSON, DOCX, and URLs.

    Example:

        kognios ingest report.pdf --db my_kb.db

        kognios ingest https://example.com --db my_kb.db --backend fts
    """
    if backend == "fts":
        from .knowledge.sqlite_fts import SQLiteKnowledge

        kb = SQLiteKnowledge(db_path=db)
    else:
        from .knowledge.numpy_vector import NumpyVectorKnowledge

        kb = NumpyVectorKnowledge()

    count = kb.load(file)
    click.echo(click.style(f"Ingested {file} → {db} ({backend}) {count} chunk(s)", fg="green"))


@cli.command("eval")
@click.argument("dataset", type=click.Path(exists=True))
@click.option(
    "--provider",
    "-p",
    default="anthropic",
    type=click.Choice(_ALL_PROVIDERS),
    help="LLM provider.",
)
@click.option("--model", "-m", default=None, help="Model name.")
@click.option(
    "--scorer",
    "-s",
    default="contains",
    type=click.Choice(["contains", "exact_match", "regex_match"]),
    show_default=True,
    help="Scoring function.",
)
@click.option(
    "--threshold",
    "-t",
    default=0.5,
    type=float,
    show_default=True,
    help="Pass threshold (0.0–1.0).",
)
def eval_cmd(dataset: str, provider: str, model: str | None, scorer: str, threshold: float):
    """Run a JSON eval dataset against a Kognios agent.

    DATASET must be a JSON file containing a list of objects with at least
    an "input" key and an optional "expected" key:

    \b
        [
          {"input": "What is 2+2?", "expected": "4"},
          {"input": "Capital of France?", "expected": "Paris"}
        ]

    Example:

        kognios eval cases.json --scorer exact_match --threshold 0.8
    """
    from .eval import AgentEvaluator, EvalCase
    from .eval.scorers import contains, exact_match, regex_match

    scorer_map = {"contains": contains, "exact_match": exact_match, "regex_match": regex_match}
    scorer_fn = scorer_map[scorer]

    with open(dataset, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise click.ClickException("Dataset must be a JSON array.")

    cases = [EvalCase(input=c["input"], expected=c.get("expected", "")) for c in raw]

    chosen_model = model or _PROVIDER_DEFAULTS[provider]
    llm = _make_model(provider, chosen_model)
    agent = Agent(model=llm)
    evaluator = AgentEvaluator(agent, scorer=scorer_fn, pass_threshold=threshold)

    click.echo(
        click.style(f"Running {len(cases)} cases with {provider}/{chosen_model} ...\n", fg="cyan")
    )
    report = evaluator.run(cases)
    report.print_report()


@cli.command()
@click.option(
    "--provider",
    "-p",
    default="anthropic",
    type=click.Choice(_ALL_PROVIDERS),
    help="LLM provider.",
)
@click.option("--model", "-m", default=None, help="Model name.")
@click.option(
    "--instructions",
    "-i",
    default="You are a helpful assistant.",
    help="System instructions.",
)
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option(
    "--kb", default=None, help="Path to a SQLite knowledge base created by `kognios ingest`."
)
def serve(
    provider: str, model: str | None, instructions: str, host: str, port: int, kb: str | None
):
    """Serve a Kognios agent over HTTP.

    Endpoints:

    \b
        POST /run                  Stateless single-turn request
        POST /stream               Stateless streaming (SSE)
        POST /session              Create a new session, returns session_id
        POST /session/{id}/run     Stateful multi-turn request
        POST /session/{id}/stream  Stateful streaming (SSE)
        DELETE /session/{id}       Delete a session
        GET /health                Health check

    Requires: pip install 'kognios[serve]'

    Example:

    \b
        kognios serve --provider anthropic --port 8000
        curl -X POST http://localhost:8000/run \\
             -H 'Content-Type: application/json' \\
             -d '{"message": "What is 2+2?"}'
    """
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError:
        raise click.ClickException("fastapi and uvicorn are required: pip install 'kognios[serve]'")

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel as PydanticBase

    chosen_model = model or _PROVIDER_DEFAULTS[provider]
    llm = _make_model(provider, chosen_model)

    knowledge = None
    if kb is not None:
        from .knowledge.sqlite_fts import SQLiteKnowledge

        knowledge = SQLiteKnowledge(db_path=kb)

    # stateless agent (no memory)
    stateless_agent = Agent(model=llm, knowledge=knowledge, instructions=instructions)

    # session store: session_id -> ShortTermMemory
    _sessions: dict[str, ShortTermMemory] = {}

    app = FastAPI(title="Kognios Agent", version="1.0")

    class RunRequest(PydanticBase):
        message: str

    class RunResponse(PydanticBase):
        response: str
        usage: dict = {}

    class SessionResponse(PydanticBase):
        session_id: str

    @app.get("/health")
    def health():
        return {"status": "ok", "provider": provider, "model": chosen_model}

    @app.post("/run", response_model=RunResponse)
    def run_endpoint(req: RunRequest):
        result = stateless_agent.run(req.message)
        return RunResponse(response=result, usage=stateless_agent.last_usage)

    @app.post("/stream")
    def stream_endpoint(req: RunRequest):
        def generate():
            for chunk in stateless_agent.stream(req.message):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/session", response_model=SessionResponse)
    def create_session():
        sid = str(uuid.uuid4())
        _sessions[sid] = ShortTermMemory(max_turns=20)
        return SessionResponse(session_id=sid)

    @app.delete("/session/{session_id}")
    def delete_session(session_id: str):
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        del _sessions[session_id]
        return {"deleted": session_id}

    @app.post("/session/{session_id}/run", response_model=RunResponse)
    def session_run(session_id: str, req: RunRequest):
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session_agent = Agent(
            model=llm,
            memory=_sessions[session_id],
            knowledge=knowledge,
            instructions=instructions,
        )
        result = session_agent.run(req.message)
        return RunResponse(response=result, usage=session_agent.last_usage)

    @app.post("/session/{session_id}/stream")
    def session_stream(session_id: str, req: RunRequest):
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session_agent = Agent(
            model=llm,
            memory=_sessions[session_id],
            knowledge=knowledge,
            instructions=instructions,
        )

        def generate():
            for chunk in session_agent.stream(req.message):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    kb_label = f" + KB {kb}" if kb else ""
    click.echo(
        click.style(
            f"Kognios serve — {provider}/{chosen_model}{kb_label} on http://{host}:{port}",
            fg="cyan",
        )
    )
    click.echo(
        click.style(
            "POST /run  POST /stream  POST /session  POST /session/{id}/run  GET /health",
            fg="bright_black",
        )
    )
    uvicorn.run(app, host=host, port=port)


def main():
    cli()


if __name__ == "__main__":
    main()
