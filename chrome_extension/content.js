/*
 * ============================================================
 * InsightCart - Amazon Review Content Script
 * ============================================================
 *
 * Purpose:
 * Extract Amazon product reviews from the currently opened page.
 *
 * Supports:
 * - Amazon.in
 * - Amazon.com
 * - Product pages
 * - Customer review pages
 * - Dynamically loaded review sections
 * - Long reviews
 * - Review title
 * - Full review body
 * - Rating
 * - Reviewer
 * - Verified purchase
 * - Review date
 * - Helpful votes
 *
 * Message:
 * {
 *     action: "extract_reviews"
 * }
 *
 * Response:
 * {
 *     product_name,
 *     category,
 *     url,
 *     total_reviews,
 *     reviews: [...]
 * }
 */

console.log("InsightCart: content.js loaded");

// ============================================================
// GLOBAL CONFIGURATION
// ============================================================

const INSIGHTCART_CONFIG = {
    MAX_REVIEWS: 100,

    // How many times to retry scrolling + searching before giving up
    MAX_RETRY_ATTEMPTS: 3,

    // Delay (ms) between retry attempts, to let Amazon lazy-load reviews
    RETRY_DELAY_MS: 700,

    SELECTORS: {
        REVIEW_CONTAINERS: [
            '[data-hook="review"]',
            '#cm-cr-dp-review-list [data-hook="review"]',
            '[id^="customer_review-"]',
            '.review',
            '.a-section.review',
            '.a-section.a-spacing-none.review',
            '[data-asin][data-hook="review"]'
        ],

        REVIEW_TITLE: [
            '[data-hook="review-title"]',
            '[data-hook="review-title"] span',
            '.review-title',
            '.a-size-base.a-link-normal.review-title',
            '.a-size-base.a-link-normal'
        ],

        REVIEW_BODY: [
            '[data-hook="review-body"]',
            '[data-hook="review-body"] span',
            '.review-text-content',
            '.review-text',
            '.a-expander-content.reviewText',
            '.a-expander-partial'
        ],

        RATING: [
            '[data-hook="review-star-rating"]',
            '[data-hook="cmps-review-star-rating"]',
            '.review-rating',
            '.a-icon-star',
            '.a-icon-alt'
        ],

        REVIEWER: [
            '.a-profile-name',
            '[data-hook="genome-widget"] .a-profile-name',
            '.author'
        ],

        DATE: [
            '[data-hook="review-date"]',
            '.review-date'
        ],

        VERIFIED: [
            '[data-hook="avp-badge"]',
            '.avp-badge',
            '.a-color-state'
        ],

        HELPFUL: [
            '[data-hook="helpful-vote-statement"]',
            '.cr-vote-buttons',
            '.helpful-votes'
        ]
    }
};


// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function cleanText(text) {
    if (!text) {
        return "";
    }

    return text
        .replace(/\u00a0/g, " ")
        .replace(/\r/g, "")
        .replace(/\t/g, " ")
        .replace(/[ ]{2,}/g, " ")
        .replace(/\n[ ]+/g, "\n")
        .replace(/[ ]+\n/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
}


function normalizeReviewText(text) {
    if (!text) {
        return "";
    }

    let result = text;

    // Remove common Amazon UI text
    result = result.replace(/\bRead more\b/gi, "");
    result = result.replace(/\bRead less\b/gi, "");

    // Remove "Helpful" UI text if accidentally captured
    result = result.replace(/^\s*Helpful\s*$/gim, "");

    // Remove "Report" UI text
    result = result.replace(/^\s*Report\s*$/gim, "");

    // Remove excessive whitespace
    result = result.replace(/\r/g, "");
    result = result.replace(/[ \t]+/g, " ");
    result = result.replace(/\n[ \t]+/g, "\n");
    result = result.replace(/[ \t]+\n/g, "\n");
    result = result.replace(/\n{3,}/g, "\n\n");

    return result.trim();
}


/*
 * ------------------------------------------------------------
 * sanitizeReviewText
 * ------------------------------------------------------------
 * Guarantees the review body always satisfies the backend's
 * schema constraints:
 *
 *     min_length = 3
 *     max_length = 5000
 *
 * This prevents 422 Unprocessable Entity errors from FastAPI
 * when a review body is empty, whitespace-only, or absurdly
 * long.
 */
function sanitizeReviewText(text) {

    let clean = (text || "").trim();

    if (clean.length < 3) {
        clean = "No review text provided.";
    }

    if (clean.length > 5000) {
        clean = clean.slice(0, 4997).trim() + "...";
    }

    return clean;
}


/*
 * ------------------------------------------------------------
 * sanitizeRating
 * ------------------------------------------------------------
 * Guarantees the rating always satisfies the backend's schema
 * constraints:
 *
 *     type = int
 *     ge = 1
 *     le = 5
 *
 * parseRating() can return floats (e.g. 4.5) or 0 when parsing
 * fails. Both of these will fail strict Pydantic v2 int
 * validation and trigger a 422 error. This function rounds and
 * clamps the value into a safe integer range.
 */
function sanitizeRating(rating) {

    const numeric = Number(rating);

    if (!Number.isFinite(numeric) || numeric <= 0) {
        return 5;
    }

    const rounded = Math.round(numeric);

    return Math.min(Math.max(rounded, 1), 5);
}


/*
 * ------------------------------------------------------------
 * sanitizeCategory
 * ------------------------------------------------------------
 * Guarantees category is never an empty string, which would
 * fail the backend's required-field validation.
 */
function sanitizeCategory(category) {

    const clean = (category || "").trim();

    return clean.length > 0 ? clean : "Electronics_5";
}


function getFirstElement(parent, selectors) {
    for (const selector of selectors) {
        try {
            const element = parent.querySelector(selector);

            if (element) {
                return element;
            }
        } catch (error) {
            console.warn(
                "InsightCart: Invalid selector:",
                selector,
                error
            );
        }
    }

    return null;
}


function getTextFromSelectors(parent, selectors) {
    const element = getFirstElement(parent, selectors);

    if (!element) {
        return "";
    }

    return cleanText(element.innerText || element.textContent || "");
}


function parseRating(text) {
    if (!text) {
        return 0;
    }

    /*
     * Examples:
     *
     * "5.0 out of 5 stars"
     * "4.0 out of 5 stars"
     * "5 out of 5 stars"
     */

    const match = text.match(/([0-5](?:\.[0-9])?)\s*out of\s*5/i);

    if (match) {
        return Number(match[1]);
    }

    const simpleMatch = text.match(/\b([1-5](?:\.[0-9])?)\b/);

    if (simpleMatch) {
        const value = Number(simpleMatch[1]);

        if (value >= 1 && value <= 5) {
            return value;
        }
    }

    return 0;
}


function parseHelpfulVotes(text) {
    if (!text) {
        return 0;
    }

    /*
     * Examples:
     *
     * "18 people found this helpful"
     * "1 person found this helpful"
     */

    const match = text.match(
        /(\d[\d,]*)\s+(?:people|person)\s+found\s+this\s+helpful/i
    );

    if (match) {
        return Number(match[1].replace(/,/g, ""));
    }

    return 0;
}


function isVerifiedPurchase(text) {
    if (!text) {
        return false;
    }

    return /verified purchase/i.test(text);
}


// ============================================================
// PRODUCT INFORMATION
// ============================================================

function extractProductName() {

    const selectors = [
        "#productTitle",
        "#title",
        "h1.product-title",
        "h1"
    ];

    for (const selector of selectors) {

        const element = document.querySelector(selector);

        if (element) {

            const text = cleanText(
                element.innerText ||
                element.textContent ||
                ""
            );

            if (text.length >= 3) {
                return text;
            }
        }
    }

    return document.title
        ? cleanText(document.title.replace(/\s*:\s*Amazon.*$/i, ""))
        : "Unknown Product";
}


// ============================================================
// PRODUCT CATEGORY
// ============================================================

function extractCategory() {

    /*
     * Amazon category extraction is not always reliable.
     *
     * We first look for breadcrumbs.
     */

    const breadcrumbSelectors = [
        "#wayfinding-breadcrumbs_container",
        "#wayfinding-breadcrumbs_feature_div",
        ".a-breadcrumb"
    ];

    for (const selector of breadcrumbSelectors) {

        const element = document.querySelector(selector);

        if (element) {

            const links = Array.from(
                element.querySelectorAll("a")
            )
                .map(link => cleanText(link.innerText))
                .filter(Boolean);

            if (links.length > 0) {

                /*
                 * The last meaningful breadcrumb is generally
                 * the most specific product category.
                 */

                const category = links[links.length - 1];

                if (category) {
                    return normalizeCategory(category);
                }
            }
        }
    }

    /*
     * Try product details / department.
     */

    const departmentSelectors = [
        "#nav-subnav .nav-a",
        "#productDetails",
        "#detailBullets_feature_div"
    ];

    for (const selector of departmentSelectors) {

        const element = document.querySelector(selector);

        if (element) {

            const text = cleanText(
                element.innerText ||
                element.textContent ||
                ""
            );

            if (/electronics/i.test(text)) {
                return "Electronics_5";
            }

            if (/books/i.test(text)) {
                return "Books_5";
            }

            if (/home/i.test(text)) {
                return "Home_5";
            }

            if (/fashion/i.test(text)) {
                return "Clothing_5";
            }
        }
    }

    /*
     * Your current model expects categories such as:
     *
     * Electronics_5
     *
     * Keep this as the default fallback for the current project.
     */

    return "Electronics_5";
}


function normalizeCategory(category) {

    if (!category) {
        return "Electronics_5";
    }

    const value = category.toLowerCase();

    if (value.includes("electronic")) {
        return "Electronics_5";
    }

    if (value.includes("book")) {
        return "Books_5";
    }

    if (
        value.includes("clothing") ||
        value.includes("fashion")
    ) {
        return "Clothing_5";
    }

    if (
        value.includes("home") ||
        value.includes("kitchen")
    ) {
        return "Home_5";
    }

    /*
     * If category is not mapped, preserve the text.
     */

    return `${category}_5`;
}


// ============================================================
// REVIEW TITLE
// ============================================================

function extractReviewTitle(reviewElement) {

    const titleElement = getFirstElement(
        reviewElement,
        INSIGHTCART_CONFIG.SELECTORS.REVIEW_TITLE
    );

    if (!titleElement) {
        return "";
    }

    let title = cleanText(
        titleElement.innerText ||
        titleElement.textContent ||
        ""
    );

    /*
     * Sometimes Amazon includes the star rating
     * together with the title.
     */

    title = title
        .replace(/^[0-5](?:\.[0-9])?\s*out of\s*5\s*stars?/i, "")
        .trim();

    title = title.replace(/^★+\s*/, "");

    return title;
}


// ============================================================
// FULL REVIEW BODY
// ============================================================

function extractReviewBody(reviewElement) {

    /*
     * IMPORTANT:
     *
     * For long reviews, don't use only:
     *
     * .innerText
     *
     * on the entire review container because that may include:
     *
     * title
     * rating
     * date
     * verified purchase
     * helpful
     * report
     *
     * Instead target Amazon's review body.
     */

    const bodyElement = getFirstElement(
        reviewElement,
        INSIGHTCART_CONFIG.SELECTORS.REVIEW_BODY
    );

    if (bodyElement) {

        let text = bodyElement.innerText ||
            bodyElement.textContent ||
            "";

        text = normalizeReviewText(text);

        if (text.length >= 3) {
            return text;
        }
    }

    /*
     * Fallback:
     *
     * Amazon sometimes changes the review body structure.
     *
     * Look for elements containing substantial text.
     */

    const possibleElements = reviewElement.querySelectorAll(
        "span, div"
    );

    let longestText = "";

    possibleElements.forEach(element => {

        const text = normalizeReviewText(
            element.innerText ||
            element.textContent ||
            ""
        );

        /*
         * Review body is generally a relatively long block.
         */

        if (
            text.length > longestText.length &&
            text.length >= 30 &&
            text.length <= 10000 &&
            !/verified purchase/i.test(text) &&
            !/people found this helpful/i.test(text)
        ) {
            longestText = text;
        }
    });

    return longestText;
}


// ============================================================
// REVIEWER
// ============================================================

function extractReviewer(reviewElement) {

    return getTextFromSelectors(
        reviewElement,
        INSIGHTCART_CONFIG.SELECTORS.REVIEWER
    );
}


// ============================================================
// REVIEW DATE
// ============================================================

function extractReviewDate(reviewElement) {

    return getTextFromSelectors(
        reviewElement,
        INSIGHTCART_CONFIG.SELECTORS.DATE
    );
}


// ============================================================
// RATING
// ============================================================

function extractReviewRating(reviewElement) {

    const ratingElement = getFirstElement(
        reviewElement,
        INSIGHTCART_CONFIG.SELECTORS.RATING
    );

    if (!ratingElement) {
        return 0;
    }

    const text =
        ratingElement.getAttribute("aria-label") ||
        ratingElement.innerText ||
        ratingElement.textContent ||
        "";

    return parseRating(text);
}


// ============================================================
// VERIFIED PURCHASE
// ============================================================

function extractVerifiedPurchase(reviewElement) {

    const text = getTextFromSelectors(
        reviewElement,
        INSIGHTCART_CONFIG.SELECTORS.VERIFIED
    );

    return isVerifiedPurchase(text);
}


// ============================================================
// HELPFUL VOTES
// ============================================================

function extractHelpfulVotes(reviewElement) {

    const text = getTextFromSelectors(
        reviewElement,
        INSIGHTCART_CONFIG.SELECTORS.HELPFUL
    );

    return parseHelpfulVotes(text);
}


// ============================================================
// VALIDATE REVIEW
// ============================================================

function isValidReview(review) {

    /*
     * A review must contain actual review body text.
     */

    if (!review.review) {
        return false;
    }

    if (review.review.length < 3) {
        return false;
    }

    /*
     * Reject obvious UI-only text.
     */

    const invalidTexts = [
        "read more",
        "read less",
        "helpful",
        "report",
        "verified purchase"
    ];

    const normalized = review.review
        .toLowerCase()
        .trim();

    if (invalidTexts.includes(normalized)) {
        return false;
    }

    return true;
}


// ============================================================
// EXTRACT SINGLE REVIEW
// ============================================================

function extractSingleReview(reviewElement, index) {

    try {

        const title = extractReviewTitle(reviewElement);

        const reviewBody = extractReviewBody(reviewElement);

        const rawRating = extractReviewRating(reviewElement);

        const reviewer = extractReviewer(reviewElement);

        const date = extractReviewDate(reviewElement);

        const verifiedPurchase =
            extractVerifiedPurchase(reviewElement);

        const helpfulVotes =
            extractHelpfulVotes(reviewElement);

        /*
         * ------------------------------------------------
         * SANITIZATION
         * ------------------------------------------------
         * These three fields (review, rating, category) are
         * sent straight to the FastAPI backend and must
         * always satisfy its Pydantic schema constraints,
         * or the entire batch request fails with a 422.
         */

        const review = {
            id: reviewElement.id || `amazon-review-${index + 1}`,

            title: title || "Untitled Review",

            review: sanitizeReviewText(reviewBody),

            rating: sanitizeRating(rawRating),

            category: sanitizeCategory(extractCategory()),

            reviewer: reviewer || "",

            date: date || "",

            verified_purchase: verifiedPurchase,

            helpful_votes: helpfulVotes
        };

        return review;

    } catch (error) {

        console.error(
            "InsightCart: Failed to extract review:",
            error
        );

        return null;
    }
}


// ============================================================
// FIND REVIEW CONTAINERS
// ============================================================

function findReviewContainers() {

    const found = [];

    /*
     * First attempt:
     * Use known Amazon review selectors.
     */

    for (
        const selector
        of INSIGHTCART_CONFIG.SELECTORS.REVIEW_CONTAINERS
    ) {

        try {

            const elements =
                document.querySelectorAll(selector);

            elements.forEach(element => {

                if (!found.includes(element)) {
                    found.push(element);
                }

            });

        } catch (error) {

            console.warn(
                "InsightCart: Selector failed:",
                selector
            );
        }
    }

    console.log(
        "InsightCart: Primary review containers:",
        found.length
    );

    /*
     * Second attempt:
     * Amazon review pages may use IDs such as:
     *
     * customer_review-Rxxxxxxxx
     */

    const idElements =
        document.querySelectorAll(
            '[id*="customer_review-"]'
        );

    idElements.forEach(element => {

        if (!found.includes(element)) {
            found.push(element);
        }

    });

    /*
     * Third attempt:
     * Search elements that contain review title/body hooks.
     */

    const bodyElements =
        document.querySelectorAll(
            '[data-hook="review-body"]'
        );

    bodyElements.forEach(body => {

        const parent =
            body.closest('[data-hook="review"]') ||
            body.closest('.review') ||
            body.closest('[id*="customer_review"]');

        if (parent && !found.includes(parent)) {
            found.push(parent);
        }

    });

    console.log(
        "InsightCart: Total candidate review containers:",
        found.length
    );

    return found;
}


// ============================================================
// FALLBACK REVIEW SEARCH
// ============================================================

function fallbackReviewSearch() {

    console.log(
        "InsightCart: Running fallback review search..."
    );

    const candidates = [];

    /*
     * Look for review body elements directly.
     */

    const bodySelectors = [
        '[data-hook="review-body"]',
        '.review-text-content',
        '.review-text',
        '.a-expander-content.reviewText'
    ];

    bodySelectors.forEach(selector => {

        document.querySelectorAll(selector)
            .forEach(element => {

                let parent =
                    element.closest('[data-hook="review"]') ||
                    element.closest('.review') ||
                    element.closest('[id*="customer_review"]');

                if (parent && !candidates.includes(parent)) {
                    candidates.push(parent);
                }
            });
    });

    /*
     * Search customer_review IDs.
     */

    document
        .querySelectorAll('[id^="customer_review-"]')
        .forEach(element => {

            if (!candidates.includes(element)) {
                candidates.push(element);
            }

        });

    console.log(
        "InsightCart: Fallback candidates:",
        candidates.length
    );

    return candidates;
}


// ============================================================
// DEDUPLICATE REVIEWS
// ============================================================

function deduplicateReviews(reviews) {

    const unique = [];
    const seen = new Set();

    for (const review of reviews) {

        if (!review || !review.review) {
            continue;
        }

        /*
         * Use title + review body as fingerprint.
         */

        const fingerprint =
            `${review.title}|${review.review}`
                .toLowerCase()
                .replace(/\s+/g, " ")
                .trim();

        if (seen.has(fingerprint)) {
            continue;
        }

        seen.add(fingerprint);

        unique.push(review);
    }

    return unique;
}


// ============================================================
// SCROLL TO REVIEWS SECTION
// ============================================================

/*
 * Amazon product pages lazy-render the reviews widget. Simply
 * querying the DOM immediately after the popup opens often
 * finds 0 review containers because Amazon hasn't fetched or
 * rendered them yet. Scrolling the reviews section into view
 * (or scrolling down generally) triggers that render.
 */
function scrollToReviews() {

    return new Promise(resolve => {

        const reviewSection = document.querySelector(
            '#reviewsMedley, #customerReviews, [data-hook="reviews-medley-widget"], #cm-cr-dp-review-list, #cm-cr-dp-review-header-a'
        );

        if (reviewSection) {
            reviewSection.scrollIntoView({
                behavior: "instant",
                block: "center"
            });
        } else {
            window.scrollTo(0, document.body.scrollHeight * 0.6);
        }

        setTimeout(resolve, 500);
    });
}


// ============================================================
// EXTRACT WITH RETRY
// ============================================================

/*
 * Retries the container search a few times with a scroll +
 * delay between attempts, since Amazon may still be loading
 * review content asynchronously right after the page appears
 * "idle".
 */
async function extractContainersWithRetry() {

    for (
        let attempt = 1;
        attempt <= INSIGHTCART_CONFIG.MAX_RETRY_ATTEMPTS;
        attempt++
    ) {

        await scrollToReviews();

        let reviewElements = findReviewContainers();

        if (reviewElements.length === 0) {
            reviewElements = fallbackReviewSearch();
        }

        if (reviewElements.length > 0) {

            console.log(
                `InsightCart: Found ${reviewElements.length} review containers on attempt ${attempt}`
            );

            return reviewElements;
        }

        console.log(
            `InsightCart: Attempt ${attempt} found 0 review containers. Retrying...`
        );

        await new Promise(
            resolve => setTimeout(
                resolve,
                INSIGHTCART_CONFIG.RETRY_DELAY_MS
            )
        );
    }

    console.warn(
        "InsightCart: No review containers found after all retry attempts."
    );

    return [];
}


// ============================================================
// MAIN EXTRACTION
// ============================================================

async function extractAmazonReviews() {

    console.log(
        "InsightCart: Starting review extraction..."
    );

    const productName = extractProductName();

    const category = sanitizeCategory(extractCategory());

    const url = window.location.href;

    console.log(
        "InsightCart: Product:",
        productName
    );

    console.log(
        "InsightCart: Category:",
        category
    );

    /*
     * --------------------------------------------------------
     * FIND REVIEW CONTAINERS (with scroll + retry)
     * --------------------------------------------------------
     */

    const reviewElements = await extractContainersWithRetry();

    let reviews = [];

    reviewElements.forEach((element, index) => {

        const review =
            extractSingleReview(element, index);

        if (review && isValidReview(review)) {
            reviews.push(review);
        }

    });

    console.log(
        `InsightCart: Extraction yielded ${reviews.length} valid reviews.`
    );

    /*
     * --------------------------------------------------------
     * DEDUPLICATE
     * --------------------------------------------------------
     */

    reviews = deduplicateReviews(reviews);

    /*
     * --------------------------------------------------------
     * LIMIT
     * --------------------------------------------------------
     */

    reviews = reviews.slice(
        0,
        INSIGHTCART_CONFIG.MAX_REVIEWS
    );

    /*
     * Ensure category is attached to every review.
     */

    reviews.forEach(review => {

        if (!review.category) {
            review.category = category;
        }

    });

    const result = {
        product_name: productName,

        category: category,

        url: url,

        total_reviews: reviews.length,

        reviews: reviews
    };

    console.log(
        "InsightCart: Amazon Data Extracted:",
        result
    );

    return result;
}


// ============================================================
// MESSAGE LISTENER
// ============================================================

chrome.runtime.onMessage.addListener(
    (message, sender, sendResponse) => {

        console.log(
            "InsightCart: Message received:",
            message
        );

        if (!message || message.action !== "extract_reviews") {
            return false;
        }

        /*
         * Async response.
         */

        extractAmazonReviews()
            .then(data => {

                sendResponse(data);

            })
            .catch(error => {

                console.error(
                    "InsightCart: Extraction failed:",
                    error
                );

                sendResponse({
                    product_name: extractProductName(),

                    category: sanitizeCategory(extractCategory()),

                    url: window.location.href,

                    total_reviews: 0,

                    reviews: [],

                    error: error.message
                });

            });

        /*
         * IMPORTANT:
         *
         * Returning true keeps the message channel open
         * while the asynchronous extraction completes.
         */

        return true;
    }
);


// ============================================================
// OPTIONAL: OBSERVE AMAZON DYNAMIC DOM
// ============================================================

let insightCartObserverTimer = null;

const insightCartObserver =
    new MutationObserver(() => {

        /*
         * Don't repeatedly extract automatically.
         *
         * We only use this observer to know that Amazon
         * changed the DOM. The actual extraction occurs
         * when popup.js requests it.
         */

        clearTimeout(insightCartObserverTimer);

        insightCartObserverTimer = setTimeout(() => {

            console.log(
                "InsightCart: Amazon page content changed."
            );

        }, 500);

    });


// Start observer
try {

    insightCartObserver.observe(
        document.body,
        {
            childList: true,
            subtree: true
        }
    );

} catch (error) {

    console.warn(
        "InsightCart: MutationObserver could not start.",
        error
    );
}


// ============================================================
// INITIAL STATUS
// ============================================================

console.log(
    "InsightCart: Amazon review extraction system ready."
);