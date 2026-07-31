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


def test_classifier_encoder_batch_failure_falls_back_to_mid():
    classifier = _build_classifier()

    class BrokenEncoder:
        identity = "broken:1"

        def encode_queries(self, texts):
            return []  # incomplete / empty batch

    classifier.encoder = BrokenEncoder()  # type: ignore[assignment]
    result = classifier.classify("summarize this in one sentence")
    assert result.tier == Tier.MID
    assert result.passed_threshold is False
    assert result.fallback_reason == "encoder/index error"
