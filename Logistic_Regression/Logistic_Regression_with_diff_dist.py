import pandas as pd
import numpy as np
from sklearn.model_selection    import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing     import normalize
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.metrics           import classification_report, accuracy_score

# 1) Load and clean the data
df = pd.read_csv("../data/cleaned_dataset.csv")
df = df.dropna(subset=["OriginalTweet", "Sentiment"])

X_text = df["OriginalTweet"].values
y      = df["Sentiment"].values

# 2) TF‑IDF without automatic normalization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    norm=None           # disable automatic normalization
)
X_tfidf = vectorizer.fit_transform(X_text)

# 3) Explicit L2 normalization
X_tfidf = normalize(X_tfidf, norm='l2', axis=1)

# 4) Filter out zero vectors
row_sums = X_tfidf.multiply(X_tfidf).sum(axis=1)       # sum of squares per row
norms    = np.sqrt(np.asarray(row_sums).reshape(-1))   # vector of norms
mask     = norms > 0

X_tfidf = X_tfidf[mask]
y       = y[mask]

print(f"Kept {X_tfidf.shape[0]} samples (dropped {np.sum(~mask)})")

# 5) Split into Train / Validation / Test sets
X_trval, X_test, y_trval, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trval, y_trval, test_size=0.25, random_state=42, stratify=y_trval
)

# 6) Define three KNN classifiers — Euclidean, Cosine, and Manhattan (all brute‑force)
knn_euc = KNeighborsClassifier(
    n_neighbors=20,
    metric='euclidean',
    algorithm='brute'
)
knn_cos = KNeighborsClassifier(
    n_neighbors=20,
    metric='cosine',
    algorithm='brute'
)
knn_manh = KNeighborsClassifier(
    n_neighbors=20,
    metric='manhattan',
    algorithm='brute'
)

# 7) Train the models
knn_euc.fit(X_train, y_train)
knn_cos.fit(X_train, y_train)
knn_manh.fit(X_train, y_train)

# 8) Evaluate on Validation and Test sets
for name, model in [
    ("Euclidean", knn_euc),
    ("Cosine",    knn_cos),
    ("Manhattan", knn_manh)
]:
    preds_val = model.predict(X_val)
    print(f"\n--- Validation ({name}) ---")
    print("Acc:", accuracy_score(y_val, preds_val))
    print(classification_report(y_val, preds_val))

    preds_test = model.predict(X_test)
    print(f"\n--- Test ({name}) ---")
    print("Acc:", accuracy_score(y_test, preds_test))
    print(classification_report(y_test, preds_test))
