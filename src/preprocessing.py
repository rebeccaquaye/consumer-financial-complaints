"""Text cleaning and keyword helpers for the complaint narratives."""
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

TEXT_COL = "Consumer complaint narrative"

# Words that appear in almost every complaint and say nothing about the product
COMPLAINT_STOPWORDS = {
    "told", "said", "called", "call", "received", "sent", "asked", "stated",
    "time", "times", "day", "days", "month", "months", "year", "years", "date",
    "company", "complaint", "issue", "help", "need", "want", "like", "just",
    "did", "does", "dont", "didnt", "im", "ive", "doesnt", "wasnt", "cant",
    "contacted", "contact", "please", "number", "information", "email", "letter",
}
STOPWORDS = list(ENGLISH_STOP_WORDS | COMPLAINT_STOPWORDS)

# Words that name each product directly (used to test for "giveaway" words)
PRODUCT_WORDS = {
    "Credit reporting": ["credit report", "credit bureau", "equifax", "experian", "transunion"],
    "Debt collection": ["debt collector", "collection agency", "collections", "collector"],
    "Credit card": ["credit card"],
    "Checking/savings": ["checking account", "savings account", "checking", "savings"],
    "Money transfer": ["paypal", "venmo", "zelle", "cash app", "wire transfer", "money transfer"],
    "Mortgage": ["mortgage", "escrow", "foreclosure"],
    "Vehicle loan": ["auto loan", "car loan", "vehicle", "car"],
    "Student loan": ["student loan", "student loans", "mohela", "nelnet", "navient"],
    "Payday/personal loan": ["payday", "personal loan", "installment loan", "title loan"],
    "Prepaid card": ["prepaid", "gift card"],
    "Debt management": ["debt settlement", "credit repair", "debt relief"],
}


def clean_text(text):
    """Lower-case a narrative and strip redactions, dollar amounts, numbers and punctuation."""
    text = text.lower()
    text = re.sub(r"x{2,}", " ", text)                 # CFPB redactions: XXXX, XX/XX/XXXX
    text = re.sub(r"\{\$[\d,.]+\}", " ", text)         # dollar amounts such as {$9400.00}
    text = re.sub(r"[^a-z\s]", " ", text)              # numbers and punctuation
    return re.sub(r"\s+", " ", text).strip()


def drop_duplicate_narratives(df):
    """Keep one copy of each narrative so template letters are not counted many times."""
    return df.drop_duplicates(subset=TEXT_COL, keep="first").reset_index(drop=True)


def top_terms(texts, n=15, ngram_range=(1, 1)):
    """Most frequent terms across all texts, as a table with counts."""
    vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=ngram_range, min_df=5)
    counts = vec.fit_transform(texts).sum(axis=0).A1
    return (pd.DataFrame({"term": vec.get_feature_names_out(), "count": counts})
              .sort_values("count", ascending=False).head(n).reset_index(drop=True))


def distinctive_terms(texts, groups, n=8):
    """For each group, the terms with the highest average TF-IDF score.

    TF-IDF down-weights words that appear in every complaint, so the top terms
    are the ones that mark out one product from the others.
    """
    vec = TfidfVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=5, max_df=0.5,
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    rows = []
    for group in sorted(set(groups)):
        mask = (groups == group).to_numpy()
        scores = X[mask].mean(axis=0).A1
        for i in scores.argsort()[::-1][:n]:
            rows.append({"product": group, "term": terms[i], "score": scores[i]})
    return pd.DataFrame(rows)


def product_word_matrix(df, text_col="clean", group_col="product_short"):
    """Share of complaints in each product (rows) that mention each product's words (columns)."""
    out = {}
    for product, words in PRODUCT_WORDS.items():
        pattern = r"\b(?:" + "|".join(words) + r")\b"
        out[product] = df[text_col].str.contains(pattern).groupby(df[group_col]).mean()
    matrix = pd.DataFrame(out) * 100
    return matrix.loc[list(PRODUCT_WORDS), list(PRODUCT_WORDS)]
