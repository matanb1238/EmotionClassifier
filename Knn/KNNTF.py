import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import numpy as np

# Step 1: Load the file
df = pd.read_csv("../data/cleaned_dataset.csv")

# Step 2: Remove rows with missing values
df = df.dropna(subset=["OriginalTweet", "Sentiment"])

# Step 3: Define input and target variables
X_text = df["OriginalTweet"]
y = df["Sentiment"]

# Step 4: Convert text to TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text)
# after X_tfidf = vectorizer.fit_transform(X_text)

# Remove zero‐vectors
row_sums = X_tfidf.multiply(X_tfidf).sum(axis=1)      # sum of squares per row
norms    = np.sqrt(np.asarray(row_sums).reshape(-1))  # L2 norms
mask     = norms > 0

X_tfidf = X_tfidf[mask]
y       = y[mask]

print(f"Kept {X_tfidf.shape[0]} samples (dropped {np.sum(~mask)})")

# Step 5: Split into Train (60%), Validation (20%), and Test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# Step 6: KNN model (with k=30, adjust as needed)
knn = KNeighborsClassifier(n_neighbors=30, metric='cosine')

knn.fit(X_train, y_train)

# Step 7: Evaluate on validation set
val_preds = knn.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# Step 8: Evaluate on test set
test_preds = knn.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))
