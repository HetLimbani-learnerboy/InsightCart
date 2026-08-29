"""
Project : InsightCart
File    : preprocessing.py
Purpose : Review preprocessing functions
"""

import re
import string

from bs4 import BeautifulSoup
import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd  # Added missing import

# Automatically download required NLTK resources if not already present
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""

    text = str(text).lower()

    # Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Reduce repeated characters (e.g., "sooo" -> "soo")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(words)
