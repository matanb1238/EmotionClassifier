import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

# --- Step 1: Load the data ---
df = pd.read_csv("../data/cleaned_dataset.csv")

# --- Step 2: Remove rows with missing values ---
df = df.dropna(subset=["OriginalTweet", "Sentiment", "TweetDay"])

# --- Step 3: Convert columns ---
df["Sentiment"] = df["Sentiment"].astype(int)
df["TweetDay"] = df["TweetDay"].astype(int)

# --- Step 4: TF-IDF on the text ---
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_text = vectorizer.fit_transform(df["OriginalTweet"])

# --- Step 5: One-hot encode the day of the week ---
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_day = encoder.fit_transform(df[["TweetDay"]])

# --- Step 6: Combine text and day-of-week features ---
X_combined = hstack([X_text, X_day])
y = df["Sentiment"]

# --- Step 7: Split into Train / Val / Test ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# --- Step 8: Logistic regression model with class balancing ---
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# --- Step 9: Validation results ---
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# --- Step 10: Test results ---
test_preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))
