import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

# --- Step 1: Load the data ---
df = pd.read_csv("../data/cleaned_Dataset_with_Tweet_Length.csv")

# --- Step 2: Clean the data ---
df = df.dropna(subset=["OriginalTweet", "Sentiment", "country"])
df["Sentiment"] = df["Sentiment"].astype(int)

# --- Step 3: Convert text to TF-IDF features ---
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_text = vectorizer.fit_transform(df["OriginalTweet"])

# --- Step 4: One-hot encode the country column ---
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_country = encoder.fit_transform(df[["country"]])

# --- Step 5: Combine text and country features ---
X_combined = hstack([X_text, X_country])
y = df["Sentiment"]

# --- Step 6: Split into Train / Validation / Test sets ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

# --- Step 7: Train the model ---
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# --- Step 8: Validate the model ---
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# --- Step 9: Test the model ---
test_preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))
