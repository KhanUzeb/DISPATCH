from src.router.classifier import Classifier
from src.router.encoder import FakeEncoder
from src.router.index import InMemoryNumpyIndex
from src.router.routes import load_route_store
from src.router.schemas import Tier


def _build_classifier():
    store = load_route_store("configs/routes.yaml", default_threshold=0.45)
    classifier = Classifier(
        route_store=store,
        encoder=FakeEncoder(),
        index=InMemoryNumpyIndex(),
        top_k=8,
        aggregation="max",
    )
    classifier.initialize()
    return classifier


def test_classifier_returns_serializable_result():
    classifier = _build_classifier()
    result = classifier.classify("summarize this in one sentence")
    assert result.classifier_version == "v4"
    assert result.route_config_version
    assert result.encoder_id
    assert result.tier == Tier.CHEAP
    assert result.passed_threshold is True


def test_classifier_routes_hard_distributed_system():
    classifier = _build_classifier()
    result = classifier.classify(
        "Design a distributed system for multi-region chat with failover, and analyze the tradeoffs"
    )
    assert result.tier == Tier.HARD
    assert result.passed_threshold is True


def test_classifier_routes_mid_explain():
    classifier = _build_classifier()
    result = classifier.classify("explain how this function works")
    assert result.tier == Tier.MID
    assert result.passed_threshold is True


def test_classifier_no_match_abstains_to_mid():
    classifier = _build_classifier()
    result = classifier.classify("   ")
    assert result.tier == Tier.MID
    assert result.passed_threshold is False
    assert result.fallback_reason


def test_classifier_routes_demo_check_prompts():
    classifier = _build_classifier()
    cheap = classifier.classify('Translate "good morning" to Spanish.')
    mid = classifier.classify("Write a short Python function that merges two sorted lists")
    hard = classifier.classify(
        "Design a fault-tolerant job queue with idempotent consumers and dead-letter handling"
    )
    assert cheap.tier == Tier.CHEAP
    assert mid.tier == Tier.MID
    assert hard.tier == Tier.HARD


def test_classifier_routes_portfolio_demo_prompts():
    classifier = _build_classifier()
    cheap = classifier.classify(
        "Summarize this in one sentence: Dispatch routes LLM requests to the right model"
    )
    mid = classifier.classify(
        "Explain the tradeoffs between REST and gRPC for an internal microservice API"
    )
    hard = classifier.classify(
        "Design a multi-region chat system with failover, consistency, and cost controls"
    )
    assert cheap.tier == Tier.CHEAP
    assert mid.tier == Tier.MID
    assert hard.tier == Tier.HARD


def test_structured_output_policy_signal_bumps_cheap():
    classifier = _build_classifier()
    result = classifier.classify("summarize this in one sentence", expects_structured_output=True)
    assert result.tier in {Tier.MID, Tier.HARD}
