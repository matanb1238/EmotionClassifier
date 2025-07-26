import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("../data/cleaned_dataset.csv")

# Step 2: Remove rows with missing values in relevant columns
df = df.dropna(subset=["OriginalTweet", "Sentiment"])

# Step 3: Input (X) and target (y) — only tweet text and sentiment
X_text = df["OriginalTweet"]
y = df["Sentiment"]

# Step 4: Convert text to TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text)

# Step 5: Split into Train (60%), Validation (20%), Test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# Step 6: Train the model with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 7: Evaluate on validation set
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# Step 8: Evaluate on test set
test_preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))
