document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultsContainer = document.getElementById("results");
    const status = document.getElementById("status");

    analyzeBtn.addEventListener("click", analyzeReviews);

    async function analyzeReviews() {
        try {
            analyzeBtn.disabled = true;
            analyzeBtn.innerText = "Analyzing...";
            status.innerText = "Extracting Amazon reviews...";

            // 1. Get active Amazon tab
            const tabs = await chrome.tabs.query({
                active: true,
                currentWindow: true
            });

            if (!tabs || tabs.length === 0) {
                throw new Error("No active browser tab found.");
            }

            const tab = tabs[0];

            if (
                !tab.url ||
                (!tab.url.includes("amazon.com") && !tab.url.includes("amazon.in"))
            ) {
                throw new Error("Please open an Amazon product page.");
            }

            // 2. Request extracted review data from content script
            let extractedData;

            try {

                extractedData = await chrome.tabs.sendMessage(
                    tab.id,
                    {
                        action: "extract_reviews"
                    }
                );

                console.log(
                    "InsightCart extracted data:",
                    extractedData
                );

            } catch (error) {

                console.error(
                    "Content script communication failed:",
                    error
                );

                throw new Error(
                    "Unable to communicate with Amazon page. Please reload the Amazon page and try again."
                );
            }

            if (
                !extractedData ||
                !extractedData.reviews ||
                extractedData.reviews.length === 0
            ) {
                const isReviewPage = tab.url.includes("/product-reviews/");
                throw new Error(
                    isReviewPage
                        ? "No reviews found on this page. Try scrolling down first."
                        : "No reviews found. Click 'See all reviews' on the product page to open the full review list, then try again."
                );
            }

            status.innerText = `Found ${extractedData.reviews.length} reviews. Analyzing...`;

            // 3. Send payload to background script to call prediction API
            const predictionResponse = await chrome.runtime.sendMessage({
                action: "predict_reviews",
                reviews: extractedData.reviews
            });

            if (!predictionResponse || !predictionResponse.success) {
                throw new Error(
                    predictionResponse?.error || "Prediction API failed."
                );
            }

            // 4. Render output in expandable review cards
            displayResults(extractedData, predictionResponse.data);
            status.innerText = "Analysis completed successfully";

        } catch (error) {
            console.error("InsightCart Error:", error);
            status.innerText = error.message;
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "Analyze Reviews";
        }
    }

    // =================================================
    // DISPLAY RESULTS
    // =================================================
    function displayResults(extractedData, data) {
        resultsContainer.innerHTML = "";

        const results = data.results || [];

        const aiCount = results.filter(item => item.prediction === 1).length;
        const humanCount = results.filter(item => item.prediction === 0).length;

        // -----------------------------------
        // SUMMARY CARD
        // -----------------------------------
        const summary = document.createElement("div");
        summary.className = "summary";
        summary.innerHTML = `
            <div class="summary-item">
                <span class="summary-number">${results.length}</span>
                <span class="summary-label">Total Reviews</span>
            </div>
            <div class="summary-item">
                <span class="summary-number" style="color: #dc2626;">${aiCount}</span>
                <span class="summary-label">AI Reviews</span>
            </div>
            <div class="summary-item">
                <span class="summary-number" style="color: #15803d;">${humanCount}</span>
                <span class="summary-label">Human Reviews</span>
            </div>
        `;

        resultsContainer.appendChild(summary);

        // -----------------------------------
        // EXPANDABLE REVIEW CARDS
        // -----------------------------------
        results.forEach((result, index) => {
            const originalReview = extractedData.reviews[index] || {};
            createReviewCard(originalReview, result);
        });
    }

    // =================================================
    // REVIEW CARD COMPONENT
    // =================================================
    function createReviewCard(review, result) {
        const card = document.createElement("div");
        card.className = "review-card";

        const title = review.title || result.title || "Untitled Review";
        const fullReview = review.review || result.clean_review || "";

        const previewLength = 180;
        let preview = fullReview;

        if (fullReview.length > previewLength) {
            preview = fullReview.substring(0, previewLength).trim() + "...";
        }

        const rating = review.rating || result.rating || 5;
        const confidence = result.confidence || "unknown";
        const reviewType = result.review_type || "Unknown";

        card.innerHTML = `
            <div class="review-header">
                <div class="review-title">
                    <span class="star">★</span> ${escapeHtml(title)}
                </div>
            </div>

            <div class="review-body">
                <span class="preview-text">${escapeHtml(preview)}</span>

                ${fullReview.length > previewLength
                ? `
                        <span class="full-text" style="display:none;">${escapeHtml(fullReview)}</span>
                        <button class="expand-btn">Show full review ▼</button>
                        `
                : ""
            }
            </div>

            <div class="prediction">
                <div class="review-type ${result.prediction === 1 ? "ai" : "human"}">
                    ${escapeHtml(reviewType)}
                </div>

                <div class="detail-row">
                    <span>Confidence</span>
                    <strong>${escapeHtml(confidence)}</strong>
                </div>

                <div class="detail-row">
                    <span>Rating</span>
                    <strong>${rating}/5</strong>
                </div>

                <div class="detail-row">
                    <span>Category</span>
                    <strong>${escapeHtml(result.category || review.category || "")}</strong>
                </div>

                ${result.reason
                ? `
                        <div class="reason">
                            <span>${escapeHtml(result.reason)}</span>
                        </div>
                        `
                : ""
            }
            </div>
        `;

        // Add toggle expand/collapse event listener
        const expandButton = card.querySelector(".expand-btn");
        if (expandButton) {
            expandButton.addEventListener("click", () => {
                const previewText = card.querySelector(".preview-text");
                const fullText = card.querySelector(".full-text");
                const isHidden = fullText.style.display === "none";

                if (isHidden) {
                    previewText.style.display = "none";
                    fullText.style.display = "inline";
                    expandButton.innerText = "Hide full review ▲";
                } else {
                    previewText.style.display = "inline";
                    fullText.style.display = "none";
                    expandButton.innerText = "Show full review ▼";
                }
            });
        }

        resultsContainer.appendChild(card);
    }

    // Utility: HTML Escaping
    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text ?? "";
        return div.innerHTML;
    }
});