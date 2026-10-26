# Consumer Financial Complaints Analysis

**Course:** INFO 4360 – Complex Data Analytics  
**University:** University of Denver, Daniels College of Business  
**Team Members:** Rebecca Quaye and Ketty Odero  
**Analysis Path:** Path A – Prediction/Classification

---

## 1. Project Overview

This project uses consumer complaint narratives from the Consumer Financial Protection Bureau (CFPB) to investigate whether the language in a complaint can predict the financial product category associated with it.

We analyze a sample of 5,000 consumer complaints across 11 financial product categories, covering August 2025 through July 2026.

## 2. Business Problem

Financial institutions receive complaints through different channels, including emails, phone calls, online forms, and customer service interactions.

Manually reviewing and categorizing these complaints can be time-consuming, especially when the volume of complaints is high.

Natural Language Processing (NLP) and machine learning could help financial institutions automatically identify the product category associated with a complaint and route it to the appropriate department.

## 3. Research Question

**Can consumer complaint narratives accurately predict the financial product category associated with a complaint?**

## 4. Analysis Approach

**Path A: Prediction/Classification**

- **Input (X):** Consumer complaint narrative
- **Target (Y):** Product (11 financial product categories)

Our approach involves exploring and cleaning complaint narratives, identifying language patterns, and developing classification models to predict financial product categories.

## 5. Dataset

**Source:** Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database

The dataset contains 5,000 consumer complaints sampled from August 2025 through July 2026.

It includes 16 variables, such as:

- Consumer complaint narrative
- Product
- Sub-product
- Issue
- Sub-issue
- Company
- Company response to consumer
- Timely response?

The sample is nearly balanced across 11 financial product categories, with approximately 454–455 complaints per category.

A detailed data dictionary is available in the `Data/` folder.

## 6. Phase 2: Exploratory Data Analysis and Text Mining

### 6.1 Exploratory Data Analysis

We explored the dataset to better understand its structure and the characteristics of consumer complaints.

Our analysis included:

- Examining missing values.
- Exploring complaint distributions across financial product categories.
- Analyzing complaint narrative lengths.
- Identifying duplicate complaint narratives.
- Reviewing sample complaints.

### 6.2 Text Preprocessing

We prepared complaint narratives for text analysis using techniques such as:

- Lowercasing text.
- Removing punctuation and unnecessary characters.
- Removing English stopwords.
- Lemmatization.
- Handling redacted information.

### 6.3 Text Mining

We explored common language patterns using:

- Word frequency analysis.
- Bigrams.
- TF-IDF.
- Word clouds.
- Product-specific word frequencies.

Some of the most common words were *account*, *credit*, *payment*, and *loan*.

We observed that some terms appear across several product categories, while others are more closely associated with specific products.

### 6.4 Sentiment Analysis

We used VADER sentiment analysis to examine the emotional tone of complaint narratives.

We compared two approaches:

**Baseline VADER:** Sentiment analysis using the original VADER lexicon.

**Domain-adjusted VADER:** An experimental approach that neutralizes selected financial terms to examine their influence on sentiment scores.

The domain adjustment is not a guaranteed improvement in accuracy. Manually labeled sentiment data would be needed to evaluate whether it improves sentiment classification.

We also compared sentiment distributions across financial product categories.

### 6.5 Preliminary Findings

Our exploratory analysis suggests that:

- The dataset is nearly balanced across 11 product categories.
- Complaint narratives contain vocabulary that may help identify financial products.
- Some financial terms overlap across product categories.
- Sentiment scores can be influenced by financial terminology.
- Further modeling is needed to evaluate how accurately complaints can be classified.

## 7. Phase 3: Planned Predictive Modeling

In Phase 3, we plan to develop and compare multiclass classification models.

### 7.1 Machine Learning Models

We plan to use TF-IDF features with:

1. Multinomial Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (SVM)

We will use a stratified 70/30 training and testing split.

### 7.2 Model Evaluation

We plan to evaluate model performance using:

- Accuracy
- Macro F1-score
- Precision and recall for each product category
- Confusion matrices

Macro F1-score will be an important metric because we want to evaluate performance across all 11 product categories.

### 7.3 Product-Word Masking

We plan to investigate whether classification models rely heavily on explicit product-related words.

For example, words such as *mortgage*, *credit card*, and *student loan* may directly reveal the associated product category.

We will compare model performance before and after masking selected product-related terms.

### 7.4 Claude API Comparison

We also plan to explore using the Claude API to classify a sample of complaint narratives.

We will compare its predictions with traditional machine learning models and examine differences in classification performance.

We will also consider privacy, cost, consistency, and scalability.

## 8. Repository Structure

```text
consumer-financial-complaints/
│
├── Data/
│   ├── cfpb_complaints_sample_2025-08_to_2026-07.csv
│   └── Data Dictionary.xlsx
│
├── Notebook/
│   ├── Ketty.ipynb
│   └── Rebecca.ipynb
│
├── src/
│   ├── data_utils.py
│   ├── make_sample.py
│   ├── preprocessing.py
│   └── text_analysis.py
│
├── .gitignore
├── index.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

### Main Project Files

- **`index.ipynb`** – Main notebook containing the combined project analysis.
- **`Notebook/Rebecca.ipynb`** – Rebecca's individual analysis notebook.
- **`Notebook/Ketty.ipynb`** – Ketty's individual analysis notebook.
- **`Data/`** – Sampled CFPB dataset and data dictionary.
- **`src/data_utils.py`** – Supporting functions for working with the dataset.
- **`src/make_sample.py`** – Script for preparing the complaint sample.
- **`src/preprocessing.py`** – Functions for cleaning and preprocessing complaint narratives.
- **`src/text_analysis.py`** – Functions supporting text mining and sentiment analysis.
- **`requirements.txt`** – Python package dependencies.
- **`.gitignore`** – Specifies files excluded from Git tracking.
- **`LICENSE`** – MIT License.

**Note:** The `.env` file stores local environment variables and API credentials. It is intentionally excluded from GitHub. The `__pycache__/` folder contains automatically generated Python cache files and does not need to be tracked.

## 9. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/rebeccaquaye/consumer-financial-complaints.git
```

### Step 2: Open the Project Folder

```bash
cd consumer-financial-complaints
```

### Step 3: Create a Python Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Open the Main Notebook

Open `index.ipynb` in VS Code or Jupyter Notebook.

Make sure the dataset is available in the `Data/` folder.

Select the appropriate Python environment and run the notebook cells in order.

The main notebook contains the exploratory analysis, text preprocessing, text mining, sentiment analysis, and planned predictive modeling approach.

## 10. Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project proposal, business problem, and research question | Completed |
| Phase 2 | Exploratory data analysis, text mining, and sentiment analysis | Developed |
| Phase 3 | Predictive modeling, model evaluation, and Claude API comparison | Planned |

## 11. License

This project is licensed under the MIT License. See the `LICENSE` file for details.