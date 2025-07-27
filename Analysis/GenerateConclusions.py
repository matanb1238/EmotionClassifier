import matplotlib.pyplot as plt
import numpy as np

models = ['Logistic Regression', 'KNN', 'SVM', 'AdaBoost', 'Random Forest', 'BERT']


bow_scores = [0.7821, 0.5184, 0.807, 0.522, 0.76, None]
tfidf_scores = [0.7688, 0.5842, 0.79, 0.521, 0.72, None]
word2vec_scores = [0.5723, 0.5472, 0.573, 0.538, 0.58, 0.8856]


x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))


def safe_bar(pos, scores, color, label):
    scores_clean = [s if s is not None else 0 for s in scores]
    bars = ax.bar(pos, scores_clean, width, label=label, color=color)
    for i, s in enumerate(scores):
        if s is not None:
            ax.text(pos[i], s + 0.01, f'{s:.2f}', ha='center', va='bottom', fontsize=8)

safe_bar(x - width, bow_scores, 'royalblue', 'BoW')
safe_bar(x, tfidf_scores, 'seagreen', 'TF-IDF')
safe_bar(x + width, word2vec_scores, 'darkorange', 'Word2Vec / BERT')


ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Comparison of Models by Vectorization Method', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right')
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.show()
