const API_BASE_URL = "http://127.0.0.1:8000/api/v1";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action !== "analyze_sentiments") {
        return;
    }

    (async () => {
        try {
            const requestUrl = `${API_BASE_URL}/sentiment/analyze/batch`;
            const reviews = message.reviews;
            if (
                !Array.isArray(reviews) ||
                reviews.length === 0 ||
                reviews.length > 40 ||
                !reviews.every(review => typeof review === "string" && review.trim())
            ) {
                sendResponse({
                    success: false,
                    error: "Provide between 1 and 40 non-empty review strings."
                });
                return;
            }

            console.info("InsightCart sentiment request", {
                url: requestUrl,
                reviewCount: reviews.length,
                reviewLengths: reviews.map(review => review.trim().length)
            });

            const response = await fetch(requestUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ reviews: reviews.map(review => review.trim()) })
            });

            // Read the response body safely.
            const data = await response.json().catch(() => null);

            if (!response.ok) {
                const detail = data?.detail;
                const apiMessage = detail?.message ||
                    (typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : null);
                console.error("Sentiment API request failed:", {
                    url: requestUrl,
                    status: response.status,
                    data
                });

                sendResponse({
                    success: false,
                    error: apiMessage || data?.message ||
                        `Sentiment API returned ${response.status}`
                });
                return;
            }

            if (!data || !Array.isArray(data.results)) {
                console.error("Unexpected sentiment API response structure", data);
                sendResponse({
                    success: false,
                    error: "Sentiment API returned an unexpected response."
                });
                return;
            }

            const results = data.results;

            const summary = results.reduce(
                (counts, item) => {
                    const label = item?.sentiment?.label;
                    if (!item?.available || !["positive", "negative", "neutral"].includes(label)) {
                        counts.failed++;
                    } else if (label === "positive") {
                        counts.positive++;
                    } else if (label === "negative") {
                        counts.negative++;
                    } else if (label === "neutral") {
                        counts.neutral++;
                    }

                    return counts;
                },
                {
                    positive: 0,
                    negative: 0,
                    neutral: 0,
                    failed: 0
                }
            );

            console.info("InsightCart sentiment analysis completed", {
                httpStatus: response.status,
                totalReviews: data.total_reviews ?? results.length,
                resultsReturned: results.length,
                summary
            });

            sendResponse({
                success: true,
                data: { ...data, summary }
            });
        } catch (error) {
            console.error("InsightCart sentiment request failed:", {
                name: error.name,
                message: error.message
            });

            sendResponse({
                success: false,
                error: error.message || "Unable to analyze sentiments."
            });
        }
    })();

    // Keep the message channel open for the asynchronous response.
    return true;
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action !== "predict_reviews") {
        return;
    }

    (async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/review/predict/batch`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    reviews: message.reviews
                })
            });

            const traceId = response.headers.get("X-MLflow-Trace-Id");
            if (!response.ok) {
                console.error(
                    "InsightCart batch prediction failed",
                    `route=/api/v1/review/predict/batch status=${response.status}`,
                    `trace_id=${traceId || "unavailable"}`
                );
                throw new Error(`API Error: ${response.status}`);
            }

            const result = await response.json();
            console.info(
                "InsightCart batch prediction completed",
                `route=/api/v1/review/predict/batch status=${response.status}`,
                `trace_id=${traceId || "unavailable"}`
            );
            sendResponse({
                success: true,
                data: result,
                traceId
            });
        } catch (error) {
            console.error("InsightCart API Error:", error);
            sendResponse({
                success: false,
                error: error.message
            });
        }
    })();

    return true;
});
