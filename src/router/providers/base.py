from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import ProviderRequest, ProviderResponse


class ProviderAdapter(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, model_id: str, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError
