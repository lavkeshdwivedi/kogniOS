from __future__ import annotations

import openai

from .openai import OpenAIModel


class AzureOpenAIModel(OpenAIModel):
    """Azure OpenAI provider.

    Authenticates using ``api_key`` or the ``AZURE_OPENAI_API_KEY`` env var.
    The ``deployment_name`` is used as the model identifier.

    Example::

        from kognios import AzureOpenAIModel
        model = AzureOpenAIModel(
            azure_endpoint="https://my-resource.openai.azure.com/",
            api_version="2024-02-01",
            deployment_name="gpt-4o",
        )
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_version: str = "2024-02-01",
        deployment_name: str = "gpt-4o",
        api_key: str | None = None,
        **kwargs,
    ):
        self.model = deployment_name
        self.kwargs = kwargs
        client_kwargs: dict = {
            "azure_endpoint": azure_endpoint,
            "api_version": api_version,
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = openai.AzureOpenAI(**client_kwargs)
        self._async_client = openai.AsyncAzureOpenAI(**client_kwargs)
