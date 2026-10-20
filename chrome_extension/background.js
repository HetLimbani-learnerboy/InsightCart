const API_BASE_URL = "http://localhost:8000/api/v1";

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