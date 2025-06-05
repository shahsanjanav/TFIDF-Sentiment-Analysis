# 💬 TF-IDF Sentiment Analysis on Twitter Dataset

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Scikit-learn](https://img.shields.io/badge/Built%20With-scikit--learn-ff69b4)

This project performs sentiment classification on Twitter data using TF-IDF vectorization and machine learning classifiers. The notebook includes data cleaning, vectorization, training, and performance evaluation using classic models like Logistic Regression and Naive Bayes.

---

## 📁 Project Structure

TFIDF-Sentiment-Analysis/
├── TF-IDFSentimentAnalysis.ipynb        		# Main notebook
├── NLP TF-IDF Sentiment Analysis.pdf        		# Project summary
├── requirements.txt                     		# Python dependencies
├── README.md                            		# Project documentation
├── .gitignore                           		# Git exclusion rules
├── sample_sentiment_dataset.csv 			# 5K sample of original dataset
└── training.1600000.processed.noemoticon.csv  	# Dataset

---

## 🧠 Dataset

The original dataset is from [Kaggle Sentiment140](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset), containing over 1.6M tweets labeled with sentiment polarity (0 = Negative, 2 = Neutral, 4 = Positive).

> ⚠️ Due to GitHub file size limits, only a 5,000-row sample is included as `sample_sentiment_dataset.csv` for testing and demo purposes.

Each record includes:
- Polarity (0/2/4)
- Tweet ID
- Date
- Query
- Username
- Cleaned tweet text

---

## 🔍 Methods Used

- Text cleaning (lowercasing, punctuation, stopwords removal)

- TF-IDF vectorization with `TfidfVectorizer`

- Machine learning classifiers:
  - Logistic Regression
  - Naive Bayes
  - SVM (optional)

- Evaluation:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix
  - Visual plots

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/TFIDF-Sentiment-Analysis.git
cd TFIDF-Sentiment-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook TF-IDFSentimentAnalysis.ipynb
```

---

## 📄 License
MIT License © 2025 Sanjana Shah

---

## 👤 Author

**Sanjana Shah**  
✨ Machine Learning & Generative AI Enthusiast  
📫 Connect on [LinkedIn](https://www.linkedin.com/in/sanjanavshah)
GitHub: [@shahsanjanav](https://github.com/shahsanjanav)

---

⭐ If you like this project, consider starring it on GitHub!