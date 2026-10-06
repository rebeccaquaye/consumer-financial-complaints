# Consumer Financial Complaints

## Project Overview

This project is part of INFO 4360: Complex Data Analytics. It uses consumer complaint narratives from the Consumer Financial Protection Bureau (CFPB) to examine whether the language in a complaint can be used to predict the financial product category associated with the complaint.

## Business Problem

Financial institutions receive large volumes of consumer complaints related to different financial products and services. Manually reviewing and categorizing these complaints can require significant time and effort. An NLP classification model could help automate the initial categorization of complaints and route them to the appropriate product teams more efficiently.

## Research Question

**Can consumer complaint narratives accurately predict the financial product category associated with a complaint?**

## Analysis Path

**Path A: Prediction/Classification**

The Consumer Complaint Narrative will be used as the primary text input (X), and Product will be the target variable (Y). NLP and machine learning techniques will be used to train and evaluate a classification model on unseen complaint narratives.

## Dataset

The data comes from the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database. The project currently uses a sample of 5,000 consumer complaint records.

Key variables include:

- Consumer complaint narrative – Primary text input (X)
- Product – Target variable (Y)
- Sub-product
- Issue
- Sub-issue
- Company
- Company response to consumer
- Timely response?

A detailed data dictionary is included in the `Data` folder.

## Repository Structure

- `Data/` – CFPB sample dataset and data dictionary
- `Python files/` – Python files used for data preparation and analysis
- `requirements.txt` – Python package requirements
