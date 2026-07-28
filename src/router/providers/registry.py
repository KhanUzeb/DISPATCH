from __future__ import annotations

from dataclasses import dataclass

from ..schemas import Provider
from .base import ProviderAdapter


@dataclass
class ProviderRegistry:
    providers: dict[Provider, ProviderAdapter]

    def get(self, provider: Provider) -> ProviderAdapter | None:
        return self.providers.get(provider)
