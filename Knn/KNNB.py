import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Step 1: Load the data ---
df = pd.read_csv("../data/cleaned_dataset.csv")

# --- Step 2: Drop missing rows ---
df = df.dropna(subset=["OriginalTweet", "Sentiment"])

# --- Step 3: Define inputs and labels ---
X_text = df["OriginalTweet"]
y      = df["Sentiment"]

# --- Step 4: Convert text to Bag‑of‑Words ---
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_bow = vectorizer.fit_transform(X_text)

# --- New: Filter out zero‑vector rows ---
row_sums = X_bow.sum(axis=1)                      # sum of counts per document
mask     = np.asarray(row_sums).reshape(-1) > 0   # keep only non‑empty docs

X_bow = X_bow[mask]
y     = y[mask]

print(f"Kept {X_bow.shape[0]} samples (dropped {np.sum(~mask)})")

# --- Step 5: Split into Train (60%), Val (20%), Test (20%) ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_bow, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# --- Step 6: Define KNN with cosine distance ---
knn = KNeighborsClassifier(n_neighbors=20, metric='cosine')
knn.fit(X_train, y_train)

# --- Step 7: Evaluate on validation set ---
val_preds = knn.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# --- Step 8: Evaluate on test set ---
test_preds = knn.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))
