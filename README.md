# 🦠 COVID-19 Tweet Sentiment Classification

Welcome to our final project for the Machine Learning course (2025).  
This project analyzes over 40,000 tweets related to the COVID-19 pandemic and classifies them into **Positive**, **Negative**, or **Neutral** sentiments.

## 📊 Goal
Automatically classify COVID-related tweets into one of three sentiment categories using a variety of machine learning models and vectorization techniques.

---

## 📁 Dataset

The dataset includes tweets with the following fields:

- `OriginalTweet`: The tweet content
- `Sentiment`: {0: Positive, 1: Negative, 2: Neutral}
- `TweetDay`: Day of week (extracted)
- `Country`: Cleaned and standardized
- `TweetLength`: Number of tokens in the tweet

> Source: [Kaggle COVID-19 Tweet Dataset](https://www.kaggle.com/datatasks/emotions)

---

## 📦 Project Structure

```bash
.
├── data/
│   ├── combined_cleaned_dataset.csv
│   └── cleaned_Dataset_with_Tweet_Length.csv
│   └── combined_cleaned_dataset.csv
├── Analysis/
│   └── GenerateConclusions.csv
├── Bert/
    └── bertclassifier.csv
├── Knn/
│   ├── KnnB.py
│   ..
├── Logistic_Regression/
│   ├── Logistic_Regression_with_days.py
│   ..
├── Models/
│   ├── ModelsTogether.py (svm, adaboost,randomforest)
│   
├── preprocessing/
│   ├── change_label_data.py
│   ├── cleanDataSet.py
│   └── diagrams.py
│   └── downloadData.py
│   └── preWork.py
├── report/
│   └── COVID_19_Sentiment Report.pdf
└── README.md
