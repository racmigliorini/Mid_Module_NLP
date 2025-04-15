# Toxic Comment Classification Challenge
### Multilabel NLP Pipeline with TF-IDF and Word Embeddings

This repository provides an end-to-end machine learning pipeline to detect multiple types of toxicity in user comments. It is based on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge).

## 🧠 Project Summary

We apply Natural Language Processing (NLP) and classic Machine Learning models to classify toxic comments across **six multilabel categories**.

### Target Labels
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

The solution includes two text representation strategies:
- TF-IDF (term frequency–inverse document frequency)
- spaCy Word Embeddings (`en_core_web_md`)

## 📁 Project Structure

- **Chapter 1** – Data loading and initial inspection
- **Chapter 2** – Exploratory Data Analysis (EDA)
- **Chapter 3** – Text preprocessing and cleaning
- **Chapter 4** – Feature extraction with TF-IDF
- **Chapter 5** – Model training and evaluation using TF-IDF
- **Chapter 6** – Feature extraction with spaCy embeddings
- **Chapter 7** – Model training using spaCy embeddings
- **Chapter 8** – Hyperparameter tuning (`RandomizedSearchCV`)
- **Chapter 9** – Final submission file generation

## ⚙️ Setup & Requirements

### Prerequisites
- Python ≥ 3.8
- pip

### Install instructions

```bash
git clone https://github.com/yourusername/toxic-comment-classification.git
cd toxic-comment-classification
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Then:

```bash
python -m spacy download en_core_web_md
```

And in Python:

```python
import nltk
nltk.download("stopwords")
```

## 📦 Input Files

Ensure the following files are present in the root folder:

- `train.csv`
- `test.csv`
- `test_labels.csv`
- `sample_submission.csv`

## 📊 Evaluation Metrics

Each model is evaluated using:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- AUC (when available)

## 📈 Models Compared

The following models were trained under both vectorization schemes:

- Logistic Regression
- Linear SVM (`SGDClassifier`)
- Decision Tree

## 📝 Final Output

- `submission.csv`: Multilabel predictions formatted for Kaggle submission
