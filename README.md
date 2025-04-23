# Toxic Comment Classification Challenge  
### Multilabel NLP Pipeline with TF-IDF and Word Embeddings

This repository provides a complete machine learning pipeline to classify user comments based on multiple types of toxicity. It is based on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge).

📁 **GitHub Repository:**  
[https://github.com/racmigliorini/Mid_Module_NLP](https://github.com/racmigliorini/Mid_Module_NLP)

---

## 🧠 Project Summary

This solution applies Natural Language Processing (NLP) techniques and classical Machine Learning models to classify toxic comments into **six non-exclusive categories**:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Two text representation strategies are compared:
- **TF-IDF** (Term Frequency-Inverse Document Frequency)
- **spaCy Word Embeddings** (`en_core_web_md`)

---

## 📁 Project Structure

- **Chapter 1** – Load and inspect the dataset
- **Chapter 2** – Exploratory Data Analysis (EDA)
- **Chapter 3** – Text cleaning and normalization
- **Chapter 4** – Feature extraction with TF-IDF
- **Chapter 5** – Model training using TF-IDF vectors
- **Chapter 6** – Feature extraction using spaCy embeddings
- **Chapter 7** – Model training using embeddings
- **Chapter 8** – Hyperparameter tuning with `RandomizedSearchCV`
- **Chapter 9** – Final submission file creation (`submission.csv`)

---

## ⚙️ Setup & Requirements

### Requirements
- Python ≥ 3.8
- pip

### Installation

```bash
git clone https://github.com/racmigliorini/Mid_Module_NLP.git
cd Mid_Module_NLP
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Then download the necessary NLP resources:

```bash
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('stopwords')"

```

And inside Python:

```python
import nltk
nltk.download("stopwords")
```

---

## 📦 Required Files

Place the following files in the project root:

- `train.csv`
- `test.csv`
- `test_labels.csv`
- `sample_submission.csv`

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1-score (macro average)
- AUC (when applicable)

---

## 🧪 Models Compared

The following classifiers were tested using both feature representations (TF-IDF and spaCy Embeddings):

- Logistic Regression
- Linear SVM (`SGDClassifier`)
- Naive Bayes (`MultinomialNB` for TF-IDF, `GaussianNB` for Embeddings)

---

## 📤 Final Output

- `submission.csv`: Multilabel predictions formatted for Kaggle submission.

---
