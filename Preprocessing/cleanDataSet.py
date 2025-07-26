import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path

# Download NLTK resources (one-time)
for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(res, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Full tweet cleaning function
def clean_tweet(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)      # Remove URLs
    text = re.sub(r"@\w+|#\w+", '', text)                    # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", '', text)                      # Remove punctuation
    text = re.sub(r"\d+", '', text)                          # Remove numbers
    tokens = word_tokenize(text, preserve_line=True)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 1
    ]
    return ' '.join(tokens)

# Function to clean a CSV file and overwrite the specified text column
def clean_dataset(csv_path: str, text_col: str = 'OriginalTweet'):
    df = pd.read_csv(csv_path, encoding='latin1')
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in the file.")
    # Remove rows with missing text
    df.dropna(subset=[text_col], inplace=True)
    # In-place cleaning of the text column
    df[text_col] = df[text_col].apply(clean_tweet)
    # Save cleaned file
    out_path = Path(csv_path).with_name(f"cleaned_{Path(csv_path).name}")
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"Cleaned file saved to: {out_path}")

# Sample execution
if __name__ == "__main__":
    clean_dataset("Dataset_with_Tweet_Length.csv")   # Cleans the 'OriginalTweet' column
