from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..schemas import ProviderRequest, ProviderResponse


@dataclass(frozen=True)
class ProviderStreamChunk:
    text: str = ""
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None


class ProviderAdapter(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError

    async def stream(self, model_id: str, request: ProviderRequest) -> AsyncIterator[str]:
        """Yield text deltas. Default: non-streaming execute as a single delta."""
        response = await self.execute(model_id, request)
        if response.text:
            yield response.text

    async def stream_with_usage(
        self, model_id: str, request: ProviderRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        async for text in self.stream(model_id, request):
            yield ProviderStreamChunk(text=text)
