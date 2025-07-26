import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import numpy as np
import string

# Step 1: Load the file
df = pd.read_csv("../data/cleaned_dataset.csv")
df = df.dropna(subset=["OriginalTweet", "Sentiment"])

# Step 2: Text cleaning and tokenization
def simple_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

df["tokens"] = df["OriginalTweet"].apply(simple_tokenize)

# Step 3: Train a Word2Vec model
w2v_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=2, workers=4)

# Step 4: Compute the average vector for each tweet
def sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_vectors = df["tokens"].apply(lambda x: sentence_vector(x, w2v_model))
X = np.vstack(X_vectors.values)

# Step 5: Encode sentiment labels
le = LabelEncoder()
y = le.fit_transform(df["Sentiment"])

# Step 6: Split into Train / Validation / Test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42
)

# Step 7: Train Logistic Regression with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 8: Evaluate on the validation set
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation Report:\n", classification_report(y_val, val_preds))

# Step 9: Evaluate on the test set
test_preds = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test Report:\n", classification_report(y_test, test_preds))
