import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import seaborn as sns
from pandas.api.types import CategoricalDtype

# -------------------------------------------------
# Step 1: Load the data (versions already include the 'Country' column)
train_df = pd.read_csv("Corona_NLP_train_numeric_with_country_v3.csv")
test_df  = pd.read_csv("Corona_NLP_test_numeric_with_country_v2.csv")

# -------------------------------------------------
# Step 2: Combine for overall analysis
df = pd.concat([train_df, test_df], ignore_index=True)

# -------------------------------------------------
# Pie chart — Sentiment
sentiment_map    = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
sentiment_counts = df['Sentiment'].value_counts().sort_index()
sentiment_labels = [sentiment_map[i] for i in sentiment_counts.index]

plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts,
        labels=sentiment_labels,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Sentiment Distribution")
plt.axis('equal')
plt.show()

# -------------------------------------------------
# Bar chart — Days of the week
ordered_days = ["Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"]
cat_type = CategoricalDtype(categories=ordered_days, ordered=True)

df['TweetAt']   = pd.to_datetime(df['TweetAt'], dayfirst=True, errors='coerce')
df['DayOfWeek'] = df['TweetAt'].dt.day_name().astype(cat_type)
day_counts      = df['DayOfWeek'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=day_counts.index, y=day_counts.values)
plt.title("Tweet Distribution by Day of the Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# Bar chart: all countries
# Ensure the column name is lowercase 'country' according to your data
country_counts = df['country'].fillna('Unknown').value_counts()

# Select the top 20 countries
top_20_countries = country_counts.head(20)

# Compute percentages of the total data
total_tweets = country_counts.sum()
percentages = (top_20_countries / total_tweets * 100).round(2)

# Append the percentages to the country names as labels
labels_with_percent = [f"{country} ({percent}%)"
                       for country, percent in zip(top_20_countries.index, percentages)]

# Plot the chart
plt.figure(figsize=(12, 6))
sns.barplot(x=labels_with_percent, y=top_20_countries.values, color='steelblue')
plt.title("Top 20 Countries by Tweet Count")
plt.xlabel("Country (Percentage of total tweets)")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


df = pd.read_csv("../data/combined_cleaned_dataset.csv")

# Map sentiment and day values to labels
sentiment_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
day_map = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
           5: 'Thursday', 6: 'Friday', 7: 'Saturday'}

df['Sentiment_Label'] = df['Sentiment'].map(sentiment_map)
df['Day_Label'] = df['TweetDay'].map(day_map)

# Remove rows with missing values
df = df.dropna(subset=['Sentiment_Label', 'Day_Label'])

# Create a combined day-sentiment column
df['Day_Sentiment'] = df['Day_Label'] + ' - ' + df['Sentiment_Label']

# Count occurrences of each combination
counts = df['Day_Sentiment'].value_counts().sort_index()

# Plot pie chart
plt.figure(figsize=(10, 10))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Sentiment by Day of the Week')
plt.axis('equal')  # Make the pie chart a perfect circle
plt.tight_layout()
plt.show()
