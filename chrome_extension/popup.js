document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("analyzeBtn");
    const resultsContainer = document.getElementById("results");
    const status = document.getElementById("status");
    let summaryResults = null;

    analyzeBtn.addEventListener("click", analyzeReviews);
    function renderSummary(data, sourceReviews) {
        summaryResults.replaceChildren();
        const summaryHeading = document.createElement("h2");
        summaryHeading.className = "inline-summary-heading";
        summaryHeading.textContent = "AI-Generated Review Summary";
        summaryResults.appendChild(summaryHeading);
        const overview = document.createElement("article");
        overview.className = "result-card overview-card";
        const summaryText = document.createElement("p");
        summaryText.className = "overall-summary";
        summaryText.textContent = data.overall_summary || "No overall summary was returned.";
        overview.appendChild(summaryText);
        const meta = document.createElement("div");
        meta.className = "summary-meta";
        meta.append(
            metadataChip("Model", data.model || "Not provided"),
            metadataChip("Generated", formatDateTime(data.generated_at))
        );
        overview.appendChild(meta);
        if (data.ai_generated !== false) {
            const badge = document.createElement("span");
            badge.className = "ai-badge";
            badge.textContent = "AI-generated";
            overview.appendChild(badge);
        }
        summaryResults.appendChild(overview);

        summaryResults.appendChild(renderFindings("Key Pros", data.pros, "positive", sourceReviews));
        summaryResults.appendChild(renderFindings("Key Cons", data.cons, "negative", sourceReviews));
        summaryResults.appendChild(renderTopics(data.key_topics));
        const sentimentNote = document.createElement("p");
        sentimentNote.className = "summary-context";
        sentimentNote.textContent = "Review-level sentiment distribution is shown in the statistics above.";
        summaryResults.appendChild(sentimentNote);
        if (typeof data.recommendation === "string" && data.recommendation.trim()) {
            const recommendation = document.createElement("article");
            recommendation.className = "result-card recommendation-card";
            const heading = document.createElement("h3"); heading.textContent = "Buying insight";
            const body = document.createElement("p"); body.textContent = data.recommendation;
            const note = document.createElement("small"); note.textContent = "A summary of submitted feedback, not a guarantee of suitability.";
            recommendation.append(heading, body, note);
            summaryResults.appendChild(recommendation);
        }
    }

    function metadataChip(label, value) {
        const item = document.createElement("div"); item.className = "meta-chip";
        const name = document.createElement("span"); name.textContent = label;
        const content = document.createElement("strong"); content.textContent = String(value);
        item.append(name, content); return item;
    }

    function renderFindings(headingText, findings, tone, sourceReviews) {
        const section = document.createElement("section"); section.className = "findings-section";
        const heading = document.createElement("h3"); heading.textContent = headingText; section.appendChild(heading);
        if (!Array.isArray(findings) || !findings.length) {
            const empty = document.createElement("p"); empty.className = "empty-note"; empty.textContent = "No findings were returned."; section.appendChild(empty); return section;
        }
        findings.forEach(finding => {
            const details = document.createElement("details"); details.className = `finding-card ${tone}`;
            const summary = document.createElement("summary");
            const topic = document.createElement("strong"); topic.textContent = finding?.topic || "Customer feedback";
            const count = document.createElement("span");
            const ids = Array.isArray(finding?.evidence_review_ids) ? finding.evidence_review_ids.filter(id => typeof id === "string") : [];
            const safeCount = Number.isFinite(finding?.evidence_count) ? finding.evidence_count : 0;
            count.textContent = `${safeCount} supporting review reference${safeCount === 1 ? "" : "s"}`;
            summary.append(topic, count); details.appendChild(summary);
            const findingText = document.createElement("p"); findingText.textContent = finding?.finding || "No details provided."; details.appendChild(findingText);
            const matched = ids.map(id => ({ id, review: sourceReviews.find(review => review.review_id === id) })).filter(item => item.review);
            if (matched.length) {
                const list = document.createElement("div"); list.className = "evidence-list";
                matched.forEach(({ id, review }) => {
                    const evidence = document.createElement("article"); evidence.className = "evidence-item";
                    const ref = document.createElement("strong"); ref.textContent = `Review ${id}`; evidence.appendChild(ref);
                    if (review.title) { const title = document.createElement("div"); title.textContent = review.title; evidence.appendChild(title); }
                    const text = document.createElement("p"); text.textContent = review.text; evidence.appendChild(text);
                    if (Number.isFinite(review.rating)) { const rating = document.createElement("small"); rating.textContent = `${review.rating}/5 stars`; evidence.appendChild(rating); }
                    list.appendChild(evidence);
                });
                details.appendChild(list);
            } else {
                const noEvidence = document.createElement("p"); noEvidence.className = "empty-note";
                noEvidence.textContent = "No linked supporting reviews are available for this finding."; details.appendChild(noEvidence);
            }
            section.appendChild(details);
        });
        return section;
    }

    function renderTopics(topics) {
        const section = document.createElement("section"); section.className = "topics-section";
        const heading = document.createElement("h3"); heading.textContent = "Key topics"; section.appendChild(heading);
        if (!Array.isArray(topics) || !topics.length) {
            const empty = document.createElement("p"); empty.className = "empty-note"; empty.textContent = "No key topics were returned."; section.appendChild(empty); return section;
        }
        const list = document.createElement("div"); list.className = "topic-list";
        topics.forEach(topic => {
            const sentiment = ["positive", "neutral", "negative", "mixed"].includes(topic?.sentiment) ? topic.sentiment : "neutral";
            const item = document.createElement("article"); item.className = `topic-card ${sentiment}`;
            const title = document.createElement("strong"); title.textContent = topic?.topic || "Topic";
            const badge = document.createElement("span"); badge.className = "topic-sentiment"; badge.textContent = sentiment;
            const finding = document.createElement("p"); finding.textContent = topic?.finding || "";
            item.append(title, badge, finding); list.appendChild(item);
        });
        section.appendChild(list); return section;
    }

    function formatDateTime(value) {
        if (typeof value !== "string" || !value) return "Not provided";
        const parsed = new Date(value);
        return Number.isNaN(parsed.getTime()) ? "Not provided" : parsed.toLocaleString();
    }

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
            resultsContainer.replaceChildren();

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
                displayNoReviewsSummary();
                status.innerText = "No reviews available to summarize.";
                return;
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

            // 4. Render the existing statistics and review cards, then summarize
            // the same selected product review set below the statistics.
            displayResults(extractedData, predictionResponse.data, sentimentResults);
            const summarySucceeded = await generateReviewSummary(extractedData);
            status.innerText = summarySucceeded
                ? "Analysis completed successfully"
                : "Review analysis completed; AI summary is unavailable.";

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

        summaryResults = document.createElement("section");
        summaryResults.className = "inline-summary";
        summaryResults.setAttribute("aria-live", "polite");
        const summaryHeading = document.createElement("h2");
        summaryHeading.className = "inline-summary-heading";
        summaryHeading.textContent = "AI-Generated Review Summary";
        const loadingMessage = document.createElement("p");
        loadingMessage.className = "summary-loading";
        loadingMessage.textContent = "Analyzing customer reviews...";
        summaryResults.append(summaryHeading, loadingMessage);
        resultsContainer.appendChild(summaryResults);

        // -----------------------------------
        // EXPANDABLE REVIEW CARDS
        // -----------------------------------
        results.forEach((result, index) => {
            const originalReview = extractedData.reviews[index] || {};
            createReviewCard(originalReview, result, sentimentResults[index]);
        });
    }

    function displayNoReviewsSummary() {
        resultsContainer.replaceChildren();
        summaryResults = document.createElement("section");
        summaryResults.className = "inline-summary empty-summary";
        const heading = document.createElement("h2");
        heading.className = "inline-summary-heading";
        heading.textContent = "AI-Generated Review Summary";
        const message = document.createElement("p");
        message.className = "empty-note";
        message.textContent = "No reviews available to summarize.";
        summaryResults.append(heading, message);
        resultsContainer.appendChild(summaryResults);
    }

    async function generateReviewSummary(extractedData) {
        const sourceReviews = (Array.isArray(extractedData.reviews) ? extractedData.reviews : [])
            .filter(review => typeof review?.review === "string" && review.review.trim())
            .map(review => {
                const normalized = { text: review.review.trim() };
                if (typeof review.id === "string" && review.id.trim() && !/^amazon-review-\d+$/i.test(review.id.trim())) {
                    normalized.review_id = review.id.trim();
                }
                if (typeof review.title === "string" && review.title.trim()) normalized.title = review.title.trim();
                if (Number.isInteger(review.rating) && review.rating >= 1 && review.rating <= 5) normalized.rating = review.rating;
                return normalized;
            });

        if (!sourceReviews.length) {
            renderSummaryError("No reviews available to summarize.");
            return false;
        }
        if (typeof extractedData.product_name !== "string" || !extractedData.product_name.trim()) {
            renderSummaryError("A product name was not available for this page.");
            return false;
        }

        const payload = {
            product_name: extractedData.product_name.trim(),
            reviews: sourceReviews
        };
        if (typeof extractedData.url === "string" && extractedData.url.trim()) payload.product_url = extractedData.url;

        try {
            const response = await sendRuntimeMessage({ action: "summarize_reviews", payload });
            if (!response?.success) {
                renderSummaryError(response?.error || "The AI summary could not be generated.", response?.retryable ? payload : null, sourceReviews);
                return false;
            }
            renderSummary(response.data, sourceReviews);
            return true;
        } catch (error) {
            renderSummaryError("Could not connect to the summary service. Check that the InsightCart backend is running.", payload, sourceReviews);
            return false;
        }
    }

    function renderSummaryError(message, retryPayload = null, sourceReviews = []) {
        if (!summaryResults) return;
        summaryResults.replaceChildren();
        const heading = document.createElement("h2");
        heading.className = "inline-summary-heading";
        heading.textContent = "AI-Generated Review Summary";
        const error = document.createElement("p");
        error.className = "summary-error";
        error.setAttribute("role", "status");
        error.textContent = message;
        summaryResults.append(heading, error);
        if (retryPayload) {
            const retry = document.createElement("button");
            retry.type = "button";
            retry.className = "summary-retry";
            retry.textContent = "Retry summary";
            retry.addEventListener("click", async () => {
                summaryResults.replaceChildren(heading, Object.assign(document.createElement("p"), {
                    className: "summary-loading",
                    textContent: "Analyzing customer reviews..."
                }));
                try {
                    const response = await sendRuntimeMessage({ action: "summarize_reviews", payload: retryPayload });
                    if (response?.success) renderSummary(response.data, sourceReviews);
                    else renderSummaryError(response?.error || "The AI summary could not be generated.", retryPayload, sourceReviews);
                } catch (_error) {
                    renderSummaryError("Could not connect to the summary service. Check that the InsightCart backend is running.", retryPayload, sourceReviews);
                }
            });
            summaryResults.appendChild(retry);
        }
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
