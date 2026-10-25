"""
Local Hugging Face sentiment inference service.

Runs CardiffNLP RoBERTa locally using PyTorch and Transformers.

This diagnostic version forces CPU inference to avoid potential
Apple MPS/Metal crashes during model inference.

No Hugging Face API token or hosted inference API is required.
"""

import asyncio
import logging
import math
import threading

from typing import Any, Dict, List, Tuple

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.config import Settings, settings


logger = logging.getLogger(__name__)


# Supported sentiment labels.
LABELS = {"negative", "neutral", "positive"}

# CardiffNLP sentiment class order.
FALLBACK_LABELS = ["negative", "neutral", "positive"]


class SentimentServiceError(Exception):
    """Controlled inference error safe to expose through the API."""

    def __init__(
        self,
        status_code: int,
        code: str,
        public_message: str,
    ):
        super().__init__(public_message)

        self.status_code = status_code
        self.code = code
        self.public_message = public_message


def preprocess_cardiffnlp_text(text: str) -> str:
    """
    Normalize mentions and URLs according to CardiffNLP conventions.
    """

    tokens = []

    for token in text.split():
        if token.startswith("@") and len(token) > 1:
            token = "@user"

        elif token.startswith("http"):
            token = "http"

        tokens.append(token)

    return " ".join(tokens)


class SentimentService:
    """
    Local sentiment analysis service.

    Features:
    - Loads the model only once per service instance.
    - Uses CPU instead of Apple's MPS for diagnostic testing.
    - Keeps the existing Hugging Face model.
    - Uses inference_mode to reduce inference overhead.
    - Serializes model inference to reduce concurrent execution risks.
    - Supports single-review and batch sentiment analysis.
    """

    def __init__(
        self,
        config: Settings = settings,
        client: Any = None,
    ):
        self.config = config

        # Retained for compatibility with the existing application.
        # The local model does not use an external inference client.
        self._client = client

        self._tokenizer = None
        self._model = None
        self._device = None

        # Protect model initialization.
        self._load_lock = threading.Lock()

        # Serialize inference calls against the shared model instance.
        self._inference_lock = threading.Lock()

    def _load_model(self) -> None:
        """
        Load the tokenizer and model once per service instance.

        IMPORTANT:
        CPU is forced temporarily to diagnose the MPS crash.
        No trained model files are deleted or replaced.
        """

        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:

            # Another thread may have loaded the model already.
            if self._model is not None and self._tokenizer is not None:
                return

            model_name = self.config.HF_SENTIMENT_MODEL

            try:
                logger.info(
                    "Loading local sentiment model: %s",
                    model_name,
                )

                # Diagnostic fix: disable MPS and use CPU.
                self._device = torch.device("cpu")

                logger.info(
                    "Using device: %s",
                    self._device,
                )

                # Load the existing Hugging Face tokenizer.
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                )

                # Load the existing pretrained sentiment model.
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                )

                # Move the model to CPU.
                model.to(self._device)

                # Set the model to evaluation mode.
                model.eval()

                # Cache only after successful initialization.
                self._tokenizer = tokenizer
                self._model = model

                logger.info(
                    "Local sentiment model loaded successfully on %s",
                    self._device,
                )

            except Exception as error:
                logger.exception(
                    "Failed to load local sentiment model: %s",
                    model_name,
                )

                # Reset partially initialized objects.
                self._tokenizer = None
                self._model = None
                self._device = None

                raise SentimentServiceError(
                    status_code=503,
                    code="sentiment_model_load_failed",
                    public_message=(
                        "The local sentiment model could not be loaded. "
                        "Check the model download and backend dependencies."
                    ),
                ) from error

    @staticmethod
    def _get_label(model: Any, index: int) -> str:
        """
        Resolve a model output index to a sentiment label.
        """

        id2label = getattr(
            model.config,
            "id2label",
            {},
        ) or {}

        raw_label = id2label.get(
            index,
            id2label.get(str(index), ""),
        )

        label = str(raw_label).strip().lower()

        if label in LABELS:
            return label

        # Fallback for CardiffNLP's three sentiment classes.
        if 0 <= index < len(FALLBACK_LABELS):
            return FALLBACK_LABELS[index]

        raise ValueError(
            f"Unsupported sentiment class index: {index}"
        )

    def _predict(self, review: str) -> Dict[str, Any]:
        """
        Perform synchronous sentiment inference.

        This method runs in a worker thread when called by analyze().
        """

        self._load_model()

        # Normalize input text.
        text = preprocess_cardiffnlp_text(review)

        # Tokenize the review.
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Ensure all input tensors are on the same device as the model.
        encoded = {
            key: value.to(self._device)
            for key, value in encoded.items()
        }

        # Serialize inference against the shared model.
        with self._inference_lock:

            with torch.inference_mode():

                # Run the sentiment model.
                output = self._model(**encoded)

                # Convert logits into probabilities.
                probabilities = torch.softmax(
                    output.logits,
                    dim=-1,
                )[0]

                # Copy probabilities to CPU before converting to Python.
                probabilities = probabilities.detach().cpu()

        scores: List[Tuple[str, float]] = []

        for index, probability in enumerate(probabilities):

            label = self._get_label(
                self._model,
                index,
            )

            confidence = float(probability.item())

            # Validate confidence scores.
            if (
                not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError(
                    "Model returned an invalid confidence score."
                )

            scores.append(
                (label, confidence)
            )

        if not scores:
            raise ValueError(
                "Model returned no sentiment scores."
            )

        # Select the label with the highest probability.
        label, confidence = max(
            scores,
            key=lambda item: item[1],
        )

        return {
            "label": label,
            "confidence": confidence,
        }

    async def analyze(
        self,
        review: str,
    ) -> Dict[str, Any]:
        """
        Analyze one review using the local sentiment model.
        """

        if not self.config.HF_SENTIMENT_ENABLED:
            raise SentimentServiceError(
                status_code=503,
                code="sentiment_disabled",
                public_message="Sentiment analysis is disabled.",
            )

        if not isinstance(review, str):
            raise SentimentServiceError(
                status_code=422,
                code="invalid_review",
                public_message="Review must be a string.",
            )

        review = review.strip()

        if not review:
            raise SentimentServiceError(
                status_code=422,
                code="invalid_review",
                public_message="Review must not be empty.",
            )

        if len(review) > self.config.HF_SENTIMENT_MAX_LENGTH:
            raise SentimentServiceError(
                status_code=422,
                code="review_too_long",
                public_message=(
                    "Review exceeds the "
                    f"{self.config.HF_SENTIMENT_MAX_LENGTH}-character "
                    "sentiment limit."
                ),
            )

        try:
            # Keep blocking tokenization and inference outside
            # FastAPI's asynchronous event loop.
            return await asyncio.to_thread(
                self._predict,
                review,
            )

        except SentimentServiceError:
            raise

        except Exception as error:
            logger.exception(
                "Local sentiment inference failed"
            )

            raise SentimentServiceError(
                status_code=503,
                code="sentiment_inference_failed",
                public_message=(
                    "Local sentiment analysis failed. "
                    "Check the backend logs."
                ),
            ) from error

    async def analyze_batch(
        self,
        reviews: List[str],
        concurrency: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple reviews while preserving their order.

        The inference lock serializes actual model execution,
        even when multiple reviews are processed concurrently.
        """

        if not self.config.HF_SENTIMENT_ENABLED:
            raise SentimentServiceError(
                status_code=503,
                code="sentiment_disabled",
                public_message="Sentiment analysis is disabled.",
            )

        semaphore = asyncio.Semaphore(
            max(1, concurrency)
        )

        async def analyze_one(
            review: str,
        ) -> Dict[str, Any]:

            async with semaphore:

                try:
                    result = await self.analyze(review)

                    return {
                        "available": True,
                        "sentiment": result,
                        "error": None,
                    }

                except SentimentServiceError as error:
                    logger.warning(
                        "Sentiment analysis unavailable for one review: %s",
                        error.code,
                    )

                    return {
                        "available": False,
                        "sentiment": None,
                        "error": error.code,
                    }

                except Exception:
                    logger.exception(
                        "Unexpected batch sentiment error"
                    )

                    return {
                        "available": False,
                        "sentiment": None,
                        "error": "sentiment_inference_failed",
                    }

        # Preserve the order of the input reviews.
        return await asyncio.gather(
            *(
                analyze_one(review)
                for review in reviews
            )
        )