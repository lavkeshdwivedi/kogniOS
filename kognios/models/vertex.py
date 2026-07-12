from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from .base import BaseModel, ModelChunk, ModelResponse


class VertexAIModel(BaseModel):
    """Google Vertex AI provider (Gemini models).

    Requires: ``pip install 'kognios[vertexai]'``

    Authentication uses Application Default Credentials:
    set ``GOOGLE_APPLICATION_CREDENTIALS`` or run
    ``gcloud auth application-default login``.

    Example::

        from kognios import VertexAIModel
        model = VertexAIModel(
            model="gemini-2.0-flash-001",
            project="my-gcp-project",
            location="us-central1",
        )
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash-001",
        project: str | None = None,
        location: str = "us-central1",
        **kwargs,
    ):
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is required: pip install 'kognios[vertexai]'"
            )
        self.model = model
        self.kwargs = kwargs
        vertexai.init(project=project, location=location)
        self._GenerativeModel = GenerativeModel
        self._instance = GenerativeModel(model)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        **call_kwargs,
    ) -> ModelResponse:
        contents = self._to_contents(messages, system)
        response = self._instance.generate_content(contents, **self.kwargs)
        text = ""
        if response.candidates:
            try:
                text = response.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                pass
        usage = {}
        if hasattr(response, "usage_metadata"):
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            }
        return ModelResponse(content=text, tool_calls=[], usage=usage)

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Iterator[ModelChunk]:
        contents = self._to_contents(messages, system)
        for chunk in self._instance.generate_content(contents, stream=True, **self.kwargs):
            text = ""
            try:
                text = chunk.candidates[0].content.parts[0].text
            except (IndexError, AttributeError):
                pass
            if text:
                yield ModelChunk(text=text)
        yield ModelChunk(tool_calls=[], final=True)

    async def acomplete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> ModelResponse:
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.complete(messages, tools, system)
        )

    async def astream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
    ) -> AsyncIterator[ModelChunk]:
        import asyncio

        chunks = await asyncio.get_event_loop().run_in_executor(
            None, lambda: list(self.stream(messages, tools, system))
        )
        for chunk in chunks:
            yield chunk

    def _to_contents(self, messages: list[dict], system: str = "") -> list:
        from vertexai.generative_models import Content, Part

        contents = []
        if system:
            contents.append(Content(role="user", parts=[Part.from_text(f"[System] {system}")]))
            contents.append(Content(role="model", parts=[Part.from_text("Understood.")]))
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(Content(role=role, parts=[Part.from_text(msg["content"])]))
        return contents
