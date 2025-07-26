import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')  # change the graphics backend


# --- Step 1: Load the file ---
df = pd.read_csv("../data/combined_cleaned_dataset.csv")

# --- Step 2: Remove rows with missing values ---
df = df.dropna(subset=["Sentiment", "TweetDay"])

# --- Step 3: Convert to float for correct mapping ---
df["TweetDay"] = df["TweetDay"].astype(float)
df["Sentiment"] = df["Sentiment"].astype(float)

# --- Step 4: Map numeric values to names ---
day_map = {
    1.0: "Sunday", 2.0: "Monday", 3.0: "Tuesday",
    4.0: "Wednesday", 5.0: "Thursday", 6.0: "Friday", 7.0: "Saturday"
}
sentiment_map = {
    0.0: "Positive",
    1.0: "Negative",
    2.0: "Neutral"
}
df["TweetDay"] = df["TweetDay"].map(day_map)
df["Sentiment"] = df["Sentiment"].map(sentiment_map)

# --- Step 5: Calculate percentages ---
sentiment_counts = df.groupby(["TweetDay", "Sentiment"]).size().unstack(fill_value=0)
sentiment_percentages = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

# --- Step 6: Order by days ---
ordered_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
sentiment_percentages = sentiment_percentages.reindex(ordered_days)

# --- Step 7: Plot pie charts ---
fig, axes = plt.subplots(1, 7, figsize=(28, 4))

for i, day in enumerate(ordered_days):
    data = sentiment_percentages.loc[day]
    axes[i].pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
    axes[i].set_title(day)

plt.tight_layout()
plt.show()

df["TweetLength"] = df["OriginalTweet"].str.len()  # based on character count
# or:
# df["TweetLength"] = df["OriginalTweet"].str.split().apply(len)  # based on word count
df.groupby("Sentiment")["TweetLength"].describe()

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df, x="Sentiment", y="TweetLength")
plt.title("Tweet Length Distribution by Sentiment")
plt.ylabel("Tweet Length (characters)")
plt.xlabel("Sentiment")
plt.show()
