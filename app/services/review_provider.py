"""Provider boundary for permitted review sources; no scraper is configured."""

from typing import Protocol

from app.schemas.review_summary import ReviewInput


class ReviewProvider(Protocol):
    async def get_reviews(self, product_id: str, limit: int) -> list[ReviewInput]: ...


class UnconfiguredReviewProvider:
    async def get_reviews(self, product_id: str, limit: int) -> list[ReviewInput]:
        raise RuntimeError("No authorized review provider is configured.")
