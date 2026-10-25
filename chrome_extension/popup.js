document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultsContainer = document.getElementById("results");
    const status = document.getElementById("status");

    analyzeBtn.addEventListener("click", analyzeReviews);

    function sendRuntimeMessage(message) {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage(message, response => {
                const runtimeError = chrome.runtime.lastError;
                if (runtimeError) {
                    reject(new Error(runtimeError.message));
                    return;
                }
                resolve(response);
            });
        });
    }

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

                console.info(
                    "InsightCart extraction completed",
                    `review_count=${extractedData?.reviews?.length || 0}`
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
            const predictionResponse = await sendRuntimeMessage({
                action: "predict_reviews",
                reviews: extractedData.reviews
            });

            if (!predictionResponse || !predictionResponse.success) {
                throw new Error(
                    predictionResponse?.error || "Prediction API failed."
                );
            }

            status.innerText = "Authenticity complete. Checking review sentiment...";
            let sentimentResults = [];
            // The extractor returns objects with a confirmed `review` field.
            // Keep those objects for authenticity; sentiment accepts strings.
            const reviewTexts = extractedData.reviews.map(item =>
                typeof item?.review === "string" ? item.review.trim() : ""
            );
            for (let offset = 0; offset < reviewTexts.length; offset += 40) {
                const chunk = reviewTexts.slice(offset, offset + 40);
                const chunkResults = [];
                try {
                    if (chunk.some(review => !review)) {
                        throw new Error("One or more extracted reviews had no review text.");
                    }

                    const sentimentResponse = await sendRuntimeMessage({
                        action: "analyze_sentiments",
                        reviews: chunk
                    });

                    if (!sentimentResponse?.success) {
                        throw new Error(sentimentResponse?.error || "Sentiment request failed.");
                    }

                    const apiResults = sentimentResponse.data?.results;
                    if (!Array.isArray(apiResults)) {
                        throw new Error("Sentiment API returned an unexpected results structure.");
                    }

                    chunk.forEach((review, index) => {
                        const item = apiResults[index];
                        const label = item?.sentiment?.label;
                        const validLabel = ["positive", "neutral", "negative"].includes(label);

                        if (item?.available && validLabel) {
                            // Keep the API's nested sentiment object intact; the UI
                            // reads item.sentiment.label and item.sentiment.confidence.
                            chunkResults.push(item);
                        } else {
                            chunkResults.push({
                                available: false,
                                sentiment: null,
                                model: item?.model ?? null,
                                error: item?.error || (item ? "invalid_sentiment_result" : "missing_result")
                            });
                        }
                    });
                } catch (error) {
                    console.warn("Sentiment batch unavailable", {
                        reviewCount: chunk.length,
                        message: error.message
                    });
                    chunkResults.push(...chunk.map(() => ({
                        available: false,
                        sentiment: null,
                        error: error.message || "sentiment_service_unavailable"
                    })));
                }
                sentimentResults.push(...chunkResults);
            }

            console.info("InsightCart sentiment results received", {
                submitted: reviewTexts.length,
                successful: sentimentResults.filter(item => item.available).length,
                failed: sentimentResults.filter(item => !item.available).length
            });

            // 4. Render output in expandable review cards
            displayResults(extractedData, predictionResponse.data, sentimentResults);
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
    function displayResults(extractedData, data, sentimentResults = []) {
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

        const sentimentCounts = sentimentResults.reduce((counts, item) => {
            const label = item?.sentiment?.label;
            if (item?.available && ["positive", "neutral", "negative"].includes(label)) {
                counts[label]++;
            } else {
                counts.failed++;
            }
            return counts;
        }, { positive: 0, neutral: 0, negative: 0, failed: 0 });

        const sentimentSummary = document.createElement("div");
        sentimentSummary.className = "summary sentiment-summary";
        sentimentSummary.setAttribute("aria-label", "Sentiment analysis summary");
        sentimentSummary.innerHTML = `
            <div class="summary-item">
                <span class="summary-number">${sentimentResults.length}</span>
                <span class="summary-label">Sentiment Analyzed</span>
            </div>
            <div class="summary-item">
                <span class="summary-number" style="color:#15803d;">${sentimentCounts.positive}</span>
                <span class="summary-label">Positive</span>
            </div>
            <div class="summary-item">
                <span class="summary-number" style="color:#dc2626;">${sentimentCounts.negative}</span>
                <span class="summary-label">Negative</span>
            </div>
            <div class="summary-item">
                <span class="summary-number" style="color:#2563eb;">${sentimentCounts.neutral}</span>
                <span class="summary-label">Neutral</span>
            </div>
            ${sentimentCounts.failed ? `<div class="summary-item"><span class="summary-number">${sentimentCounts.failed}</span><span class="summary-label">Unavailable</span></div>` : ""}
        `;
        resultsContainer.appendChild(sentimentSummary);

        // -----------------------------------
        // EXPANDABLE REVIEW CARDS
        // -----------------------------------
        results.forEach((result, index) => {
            const originalReview = extractedData.reviews[index] || {};
            createReviewCard(originalReview, result, sentimentResults[index]);
        });
    }

    // =================================================
    // REVIEW CARD COMPONENT
    // =================================================
    function createReviewCard(review, result, sentimentResult) {
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
        const sentiment = sentimentResult?.available ? sentimentResult.sentiment : null;
        const sentimentLabel = ["positive", "neutral", "negative"].includes(sentiment?.label)
            ? sentiment.label
            : null;
        const sentimentConfidence = Number(sentiment?.confidence);
        const sentimentModel = sentimentResult?.model?.name;
        const sentimentError = sentimentResult?.error;

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
                <section class="sentiment-result ${sentimentLabel || "unavailable"}" aria-label="Sentiment analysis">
                    <div class="sentiment-heading">Sentiment <span>(separate from authenticity)</span></div>
                    ${sentimentLabel
                ? `<div class="sentiment-label">${escapeHtml(sentimentLabel)}</div>
                       ${Number.isFinite(sentimentConfidence) ? `<div class="sentiment-confidence">Confidence: ${Math.round(sentimentConfidence * 100)}%</div>` : ""}
                       ${sentimentModel ? `<div class="sentiment-confidence">Model: ${escapeHtml(sentimentModel)}</div>` : ""}`
                : `<div class="sentiment-label">Unavailable</div>
                       <div class="sentiment-confidence">${escapeHtml(sentimentError || "Sentiment service did not return a result.")}</div>`
            }
                    <div class="sentiment-note">Opinion prediction only; it does not prove a review is genuine or fake.</div>
                </section>
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
