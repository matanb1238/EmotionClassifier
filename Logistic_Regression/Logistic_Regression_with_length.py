import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1) Load the dataset
df = pd.read_csv("../data/cleaned_dataset.csv", encoding='latin1')

# 2) Drop rows with missing values in relevant columns
df = df.dropna(subset=["OriginalTweet", "Sentiment", "TweetLength"])

# Ensure TweetLength is numeric
df["TweetLength"] = df["TweetLength"].astype(float)

# 3) Define input features (X) and target labels (y)
X = df[["OriginalTweet", "TweetLength"]]
y = df["Sentiment"]

# 4) Create transformation pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english"), "OriginalTweet"),
        ("scale_length", StandardScaler(), ["TweetLength"])
    ]
)

# 5) Combine preprocessing and classifier into a pipeline
model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# 6) Split data into Train+Validation and Test (80% / 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Then split Train+Val into Train and Validation (80% / 20% of 80%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
)

# 7) Train the model
model.fit(X_train, y_train)

# 8) Evaluate on the validation set
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds, digits=4))

# 9) Evaluate on the test set
test_preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds, digits=4))
