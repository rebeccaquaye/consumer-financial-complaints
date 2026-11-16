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

## 7. Phase 3: Predictive Modeling

We developed and evaluated multiclass classification models to predict financial product categories from consumer complaint narratives.

### 7.1 Data Preparation

After removing exact duplicate narratives, 4,895 unique complaints remained.

We used a stratified 70/30 train-test split:

- Training set: 3,426 complaints
- Test set: 1,469 complaints
- Random state: 42

### 7.2 Machine Learning Models

We trained and compared the following models:

1. Multinomial Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (SVM)

These models used TF-IDF features extracted from complaint narratives.

We also evaluated:

- Sentence-transformer similarity using MiniLM
- spaCy semantic similarity
- Claude zero-shot classification
- Majority-class baseline

### 7.3 Model Evaluation

We evaluated classification performance using:

- Accuracy
- Macro F1-score
- Precision and recall by product category
- Confusion matrices
- Five-fold cross-validation on the training data

Macro F1-score was particularly important for evaluating performance across all 11 financial product categories.

### 7.4 Additional Experiments

We also conducted experiments involving:

- Product-word masking to assess potential information leakage.
- SVM decision-margin analysis to examine prediction confidence.
- Subgroup analysis to investigate differences in model performance.
- Claude API comparisons to evaluate LLM classification performance.

### 7.5 Model Performance Results

We compared eight classification approaches using the held-out test set.

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Linear SVM | 73.8% | 0.736 |
| Logistic Regression | 72.7% | 0.727 |
| Claude Zero-Shot | 72.6% | 0.724 |
| Multinomial Naive Bayes | 70.3% | 0.697 |
| MiniLM (Complaint-Informed) | 48.7% | 0.474 |
| MiniLM (Label-Only) | 43.5% | 0.411 |
| spaCy Similarity | 21.4% | 0.177 |
| Majority-Class Baseline | 9.3% | 0.016 |

**Best-performing model:** Linear SVM, achieving 73.8% accuracy and a macro F1-score of 0.736.

These results suggest that traditional machine learning models using TF-IDF features can perform competitively with large language models on specialized text classification tasks.

### 7.6 Product-Word Masking and Robustness Testing

We investigated whether the classification models relied heavily on explicit financial product terms.

We masked 15 product-identifying phrases, including terms such as *mortgage*, *credit card*, and *student loan*.

The Linear SVM results were:

| Evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| Original Narratives | 73.8% | 0.736 |
| Masked Narratives | 72.2% | 0.721 |

Masking changed the model's predictions for 677 of the 1,469 test complaints.

The relatively small decrease in performance suggests that the model learns additional contextual information beyond explicit product names.

### 7.7 Confidence-Aware Complaint Routing

We examined Linear SVM decision margins to understand how prediction confidence relates to classification accuracy.

- Lowest decision-margin quintile: 43.5% accuracy.
- Highest decision-margin quintile: 98.0% accuracy.

Complaints with smaller decision margins were more likely to be misclassified.

These results suggest that a confidence-aware routing system could automatically process higher-confidence complaints while directing uncertain predictions to human reviewers.

**Important:** SVM decision margins are not calibrated probabilities and should not be interpreted as direct probabilities of correct classification.

### 7.8 Subgroup Analysis

We examined classification performance across financial product categories to identify areas where the model performed well and where improvements were needed.

The analysis showed that some financial products were more difficult to distinguish because their complaint narratives contained overlapping terminology.

In particular, complaints involving debt and credit-related products presented classification challenges.

These findings highlight the importance of evaluating per-category precision, recall, and F1-scores rather than relying solely on overall accuracy.

### 7.9 Claude API Comparison and Considerations

We evaluated Claude as a zero-shot classifier and compared its performance with traditional machine learning models.

Claude achieved:

- Test accuracy: 72.6%
- Macro F1-score: 0.724

Linear SVM performed slightly better, achieving 73.8% accuracy and a macro F1-score of 0.736.

We also considered the following practical factors:

- **Privacy:** Complaint narratives may contain sensitive consumer information.
- **Cost:** API-based classification introduces usage costs.
- **Consistency:** Model predictions should be evaluated for stability.
- **Scalability:** Traditional machine learning models may be more economical for processing large complaint volumes.

The recorded Claude API usage cost was approximately $1.14 for the logged portion of the experiments. This does not represent a complete production cost estimate.

### 7.10 Business Recommendations

Based on our findings, we recommend using Linear SVM as the initial model for complaint classification.

The proposed complaint-routing process would:

1. Receive a consumer complaint narrative.
2. Preprocess the text and generate TF-IDF features.
3. Predict the financial product category using Linear SVM.
4. Route high-confidence predictions to the appropriate department.
5. Send uncertain predictions to human reviewers.

This approach could reduce manual classification effort while maintaining human oversight.

However, the model should undergo further testing using real-world complaint distributions before operational deployment.

### 7.11 Limitations and Future Work

Several limitations should be considered:

- The dataset was sampled to be nearly balanced across 11 product categories.
- Real-world complaint volumes may have different category distributions.
- Some financial product categories share similar terminology.
- Model performance may change over time as complaint language evolves.
- The proposed routing process has not been validated in a live operational environment.

Future work could include probability calibration, additional model testing, and monitoring classification performance over time.

consumer-financial-complaints/
│
├── Data/
│   ├── cfpb_complaints_sample_2025-08_to_2026-07.csv
│   └── Data Dictionary.xlsx
│
├── Notebook/
│   ├── Rebecca_EDA.ipynb
│   ├── Rebecca_Modelling.ipynb
│   ├── Rebecca_Final_CFPB_Consumer_Complaints.ipynb
│   ├── Ketty_EDA.ipynb
│   ├── Ketty_Modelling.ipynb
│   └── Ketty_Final_CFPB_Consumer_Complaints.ipynb
│
├── outputs/
│   ├── rebecca_modelling/
│   └── unified_analysis/
│
├── src/
│   └── Supporting Python scripts
│
├── .gitignore
├── Final_Index.ipynb
├── LICENSE
├── README.md
└── requirements.txt

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

### Step 5: Install the spaCy Language Model

Some semantic similarity experiments require the spaCy English language model.

```bash
python -m spacy download en_core_web_md
```

### Step 6: Open the Main Notebook

Open **`Final_Index.ipynb`** in VS Code or Jupyter Notebook.

Ensure that:

- The CFPB dataset is available in the `Data/` folder.
- The appropriate Python environment is selected.
- All required dependencies are installed.
- Any API credentials needed for optional Claude experiments are configured securely.

Run the notebook cells in order.

### Step 7: Review the Results

The main notebook includes:

- Exploratory data analysis.
- Text preprocessing and text mining.
- Sentiment analysis.
- Multiclass complaint classification.
- Model performance comparisons.
- Product-word masking experiments.
- Confidence analysis.
- Business recommendations and conclusions.

Additional modeling notebooks are available in the `Notebook/` folder.

Saved evaluation results and visualizations are available in the `outputs/` folder.

**Reproducibility Note:** Some Claude API experiments may require valid API credentials and incur usage costs. Previously saved outputs can be reviewed without rerunning those API requests.


## 10. Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project proposal, business problem, and research question | Completed |
| Phase 2 | Exploratory data analysis, text mining, and sentiment analysis | Completed |
| Phase 3 | Predictive modeling, model evaluation, and Claude API comparison | Completed |
| Phase 4 | Robustness testing, confidence analysis, and business recommendations | Completed |

## 11. License

This project is licensed under the MIT License. See the `LICENSE` file for details.