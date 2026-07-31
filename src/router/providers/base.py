from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..schemas import ProviderRequest, ProviderResponse


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
