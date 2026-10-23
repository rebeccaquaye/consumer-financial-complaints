"""Sentiment scoring and topic modeling for the complaint narratives."""
import re
from collections import Counter

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.preprocessing import STOPWORDS

# Words VADER treats as positive emotions but that are just financial topic words
# here ("credit report", "interest rate", "Fair Credit Reporting Act", "Wells Fargo").
FINANCE_WORDS = ["credit", "interest", "fair", "agreement", "original", "wells", "party",
                 "clear", "support", "supporting", "trust", "balance", "save", "savings",
                 "gift", "secure", "security", "reward", "rewards", "free", "benefit",
                 "benefits", "legal", "pay"]


def _for_vader(text):
    """VADER uses capitals and punctuation, so only remove redactions and dollar amounts."""
    text = re.sub(r"X{2,}", " ", text)
    return re.sub(r"\{\$[\d,.]+\}", " ", text)


def sentiment_scores(texts, neutral_words=None):
    """VADER compound score per text (-1 very negative to +1 very positive).

    neutral_words: words to set to 0 in VADER's dictionary before scoring.
    """
    analyzer = SentimentIntensityAnalyzer()
    for word in neutral_words or []:
        analyzer.lexicon.pop(word, None)
    return pd.Series([analyzer.polarity_scores(_for_vader(t))["compound"] for t in texts],
                     index=texts.index)


def sentiment_label(scores):
    """Standard VADER cut-offs: below -0.05 negative, above +0.05 positive."""
    return pd.cut(scores, [-1.01, -0.05, 0.05, 1.01], labels=["Negative", "Neutral", "Positive"])


def positive_word_contributions(clean_texts, n=12):
    """Which positive words add the most to VADER scores across all texts."""
    lexicon = SentimentIntensityAnalyzer().lexicon
    totals = Counter()
    for text in clean_texts:
        for word in text.split():
            value = lexicon.get(word, 0)
            if value > 0:
                totals[word] += value
    return pd.DataFrame(totals.most_common(n), columns=["word", "positive_points"])


def fit_topics(clean_texts, n_topics=10, n_words=8):
    """NMF topic model on TF-IDF features.

    Returns (topic_words, doc_topic): the top words of each topic, and the
    main topic of every document.
    """
    vec = TfidfVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=10, max_df=0.4,
                          sublinear_tf=True)
    X = vec.fit_transform(clean_texts)
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
    weights = nmf.fit_transform(X)
    terms = vec.get_feature_names_out()
    topic_words = pd.DataFrame({
        "topic": range(n_topics),
        "top_words": [", ".join(terms[row.argsort()[::-1][:n_words]]) for row in nmf.components_],
    })
    doc_topic = pd.Series(weights.argmax(axis=1), index=clean_texts.index)
    return topic_words, doc_topic
