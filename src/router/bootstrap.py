"""Shared RoutingService construction for API and MCP entrypoints."""

from __future__ import annotations

import logging
import os

import httpx

from .classifier import Classifier
from .clients import load_client_config
from .config import ConfigError, load_config
from .encoder import FakeEncoder, HuggingFaceEncoder
from .executor import Executor
from .index import InMemoryNumpyIndex
from .models import load_model_registry
from .providers.factory import build_provider_registry
from .routes import load_route_store
from .service import RoutingService

logger = logging.getLogger(__name__)


async def build_routing_service(
    *,
    http_client: httpx.AsyncClient | None = None,
    log_startup: bool = True,
) -> tuple[RoutingService | None, str | None]:
    """Build a ready RoutingService, or return (None, error)."""
    try:
        cfg = load_config()
        route_store = load_route_store(cfg.routes_path, default_threshold=cfg.default_threshold)
        model_registry = await load_model_registry(cfg.models_path, client=http_client)
        client_config = load_client_config(cfg.clients_path)
        if os.environ.get("ROUTER_USE_FAKE_ENCODER", "").lower() == "true":
            encoder = FakeEncoder()
            if log_startup:
                # stderr via logging — stdout must stay clean for MCP stdio JSON-RPC
                logger.info(
                    "encoder=fake (set ROUTER_USE_FAKE_ENCODER=false for real MiniLM routing)"
                )
        else:
            try:
                encoder = HuggingFaceEncoder(cfg.encoder_model, device=cfg.encoder_device)
                encoder.warmup()
                if log_startup:
                    logger.info("encoder=%s", cfg.encoder_model)
            except Exception as exc:
                if log_startup:
                    logger.warning(
                        "real encoder unavailable (%s); falling back to FakeEncoder",
                        exc,
                    )
                encoder = FakeEncoder()

        classifier = Classifier(
            route_store=route_store,
            encoder=encoder,
            index=InMemoryNumpyIndex(),
            top_k=cfg.top_k,
            aggregation=cfg.route_aggregation,
        )
        classifier.initialize()
        providers = build_provider_registry(timeouts_ms=cfg.provider_timeouts_ms)
        service = RoutingService(
            classifier,
            Executor(
                providers,
                max_fallbacks=cfg.max_fallbacks,
                retries_by_provider=cfg.provider_retries,
            ),
            model_registry,
            cfg,
            client_config=client_config,
        )
        return service, None
    except ConfigError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"startup failed: {exc}"
