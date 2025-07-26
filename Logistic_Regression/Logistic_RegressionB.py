import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("../data/cleaned_dataset.csv")



# 3. Handle missing values
df = df.dropna(subset=['OriginalTweet', 'Sentiment'])

# 4. Define features and labels
X_text = df['OriginalTweet']
y = df['Sentiment']

# 5. Convert text to Bag-of-Words
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_bow = vectorizer.fit_transform(X_text)

# 6. Split into train (60%), validation (20%), test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X_bow, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
# 0.25 * 0.8 = 0.2 → 60/20/20 split

# 7. Train Logistic Regression with class_weight balanced
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 8. Evaluate on validation set
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_acc:.4f}")

# 9. Evaluate on test set
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive', 'Negative', 'Neutral'],
            yticklabels=['Positive', 'Negative', 'Neutral'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig("confusion_matrix_bow.png")
plt.close()
