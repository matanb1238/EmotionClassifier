import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the data
df = pd.read_csv("../data/cleaned_dataset.csv")

# Remove rows with missing values
df = df.dropna(subset=["OriginalTweet", "Sentiment"])

# Define variables
X_text = df["OriginalTweet"]
y = df["Sentiment"]

# Initial split: 80% Train/Validation, 20% Test
X_trainval_text, X_test_text, y_trainval, y_test = train_test_split(
    X_text, y, test_size=0.2, stratify=y, random_state=42
)

# Further split: of 80%, split into 75% Train and 25% Validation → total 60% Train, 20% Validation
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_trainval_text, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
)

# Convert text to TF-IDF with max_df filtering words that appear in more than 80% of documents
vectorizer = TfidfVectorizer(max_df=0.8, ngram_range=(1, 1), stop_words="english")
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

# Build the model
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', class_weight='balanced')
model.fit(X_train, y_train)

# Predict on validation set
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:")
print(classification_report(y_val, val_preds))

# Predict on test set
test_preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:")
print(classification_report(y_test, test_preds))
