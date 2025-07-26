import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import string

# Step 1: Load and clean the dataset
df = pd.read_csv("../data/cleaned_dataset.csv")
df = df.dropna(subset=["OriginalTweet", "Sentiment"])

# Step 2: Basic text cleaning and tokenization
def simple_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

df["tokens"] = df["OriginalTweet"].apply(simple_tokenize)

# Step 3: Train a Word2Vec model on the tokenized tweets
w2v_model = Word2Vec(
    sentences=df["tokens"],
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Step 4: Compute the average Word2Vec vector for each tweet
def sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        # Return a zero vector if none of the tokens are in the vocabulary
        return np.zeros(model.vector_size)

X_vectors = df["tokens"].apply(lambda tokens: sentence_vector(tokens, w2v_model))
X = np.vstack(X_vectors.values)

# Step 5: Filter out tweets with zero vectors (no words in the Word2Vec vocabulary)
norms = np.linalg.norm(X, axis=1)
mask = norms > 0
X = X[mask]
df = df[mask]  # Align the DataFrame with filtered X

print(f"Kept {X.shape[0]} samples (dropped {np.sum(~mask)})")

# Step 6: Encode sentiment labels as integers
le = LabelEncoder()
y = le.fit_transform(df["Sentiment"])

# Step 7: Split into Train/Validation/Test sets (60%/20%/20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)

# Step 8: Train KNN classifier using cosine distance on the sentence embeddings
knn = KNeighborsClassifier(
    n_neighbors=7,
    metric='cosine',
    algorithm='brute'
)
knn.fit(X_train, y_train)

# Step 9: Evaluate on validation and test sets
print("Validation Accuracy:", accuracy_score(y_val, knn.predict(X_val)))
print("Validation Report:\n", classification_report(y_val, knn.predict(X_val)))

print("Test Accuracy:", accuracy_score(y_test, knn.predict(X_test)))
print("Test Report:\n", classification_report(y_test, knn.predict(X_test)))
