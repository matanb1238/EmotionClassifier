# Emotion Classification from Tweets 🧠💬

This project analyzes a dataset of ~400,000 tweets labeled by emotion  
and trains machine learning models to classify emotions based on text content.

## 📂 Dataset
The dataset contains short English tweets, each labeled with one of the following emotions:
- **sadness**
- **joy**
- **love**
- **anger**
- **fear**
- **surprise**

Each row includes:
- `text`: the tweet
- `label`: numeric class of the emotions (0–5)

## 🎯 Project Goals
- Analyze the dataset and understand patterns in emotion-related language
- Answer research questions like:
  - Which words are typical for each emotion?
  - Which emotions are commonly confused?
  - Do certain emotions appear in shorter or longer tweets?

## 🧪 Models
We implemented several models:
- **TF-IDF + SVM** (primary)
- **TF-IDF + Logistic Regression** (partner’s part)
- Future option: **BERT / CNN** for deep learning

## 📊 Results (SVM example)
- Accuracy: **89%**
- Strong performance on **joy**, **sadness**
- Weakest performance on **love**, **surprise**
- Confusion matrix and PCA visualizations included

## 🧩 Visualizations
- Emotion distribution barplot
- Tweet length per emotion (boxplot)
- Confusion matrix heatmap
- PCA projection of test tweets with prediction correctness
- Sample predictions (true vs. predicted)

## 👥 Authors
- Matan Blaich
- Oriya Tzabari  
(Ariel University - Machine Learning Course)

## 📝 Instructions
Open the main notebook in [Google Colab](https://colab.research.google.com/) to explore, train, and visualize models:

- `notebooks/svm_model.ipynb`

---

