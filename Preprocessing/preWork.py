import pandas as pd
#1-negative
#0-positive
#2-natural
# קריאת הקבצים עם קידוד תואם
train_df = pd.read_csv("Corona_NLP_train.csv", encoding="ISO-8859-1")
test_df = pd.read_csv("Corona_NLP_test.csv", encoding="ISO-8859-1")

# מיפוי רגשות למספרים
sentiment_mapping = {
    "Positive": 0,
    "Extremely Positive": 0,
    "Negative": 1,
    "Extremely Negative": 1,
    "Neutral": 2
}

train_df["Sentiment"] = train_df["Sentiment"].map(sentiment_mapping)
test_df["Sentiment"] = test_df["Sentiment"].map(sentiment_mapping)

# המרה של תאריכים ליום בשבוע (1=ראשון, 7=שבת)
def convert_to_weekday(date_str):
    try:
        day = pd.to_datetime(date_str, dayfirst=True).dayofweek  # 0=שני ... 6=ראשון
        return (day + 1) % 7 + 1  # שנה ל-1=ראשון ... 7=שבת
    except:
        return None

train_df["TweetDay"] = train_df["TweetAt"].apply(convert_to_weekday)
test_df["TweetDay"] = test_df["TweetAt"].apply(convert_to_weekday)

# שמירת הקבצים החדשים
train_df.to_csv("Corona_NLP_train_numeric.csv", index=False, encoding="utf-8")
test_df.to_csv("Corona_NLP_test_numeric.csv", index=False, encoding="utf-8")

print("✔ העמודות עודכנו ונשמרו בהצלחה כולל המרת רגשות ויום בשבוע")
