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

            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }

            const result = await response.json();
            sendResponse({
                success: true,
                data: result
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