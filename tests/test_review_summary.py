import asyncio
from types import SimpleNamespace

from app.schemas.review_summary import ReviewInput, SummarizeReviewsRequest
from app.services.review_summary_service import ReviewSummaryService, _batches, prepare_reviews


def test_preprocessing_deduplicates_by_id_and_normalized_text():
    rows = [
        ReviewInput(review_id="a", text="  Very   comfortable "),
        ReviewInput(review_id="a", text="Changed duplicate ID"),
        ReviewInput(text="very comfortable"),
    ]
    cleaned = prepare_reviews(rows)
    assert len(cleaned) == 2
    assert cleaned[0].text == "Very comfortable"


def test_batches_respect_configured_token_estimate():
    reviews = [ReviewInput(text="x" * 100), ReviewInput(text="y" * 100)]
    batches = _batches(reviews, token_limit=60)
    assert len(batches) == 2
    assert all(sum((len(row.text) + 3) // 4 + 32 for row in batch) <= 60 for batch in batches)


def test_summary_filters_unrecognized_evidence_ids_and_computes_counts():
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            content = (
                '{"overall_summary":"Comfortable, with limited evidence.",'
                '"pros":[{"topic":"Comfort","finding":"One reviewer liked comfort.","evidence_review_ids":["r1","invented"]}],'
                '"cons":[],"key_topics":[],"recommendation":"Consider the small sample."}'
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    config = SimpleNamespace(
        GROQ_API_KEY="", GROQ_MODEL="mock-model", GROQ_TIMEOUT=1,
        REVIEW_SUMMARY_MAX_REVIEWS=100, REVIEW_SUMMARY_BATCH_TOKENS=1000,
        REVIEW_SUMMARY_CACHE_TTL_HOURS=24, HF_SENTIMENT_ENABLED=False,
        HF_SENTIMENT_MAX_LENGTH=512,
    )
    request = SummarizeReviewsRequest(
        product_id="p1", product_name="Headphones",
        reviews=[ReviewInput(review_id="r1", text="Very comfortable")],
    )
    result = asyncio.run(ReviewSummaryService(client=client, settings_obj=config).summarize(request))
    assert result.pros[0].evidence_review_ids == ["r1"]
    assert result.pros[0].evidence_count == 1
    assert result.sentiment.analyzed == 0
    assert result.sentiment.failed == 0
