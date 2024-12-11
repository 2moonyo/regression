# regression
Immo Regression
Project Structure

The project is divided into four main classes:

Preprocessing: Cleans and merges income, zipcode, and property datasets.

FeatureEngineering: Adds derived features like region and removes outliers.

ModelApply: Handles model training and cross-validation using CatBoost.

ModelEvaluation: Computes metrics, visualizes results, and performs SHAP analysis.

Prerequisites

Python 3.7+

Libraries:

pandas

numpy

sklearn

seaborn

matplotlib

CatBoost

shap

rapidfuzz

scipy

Please refer to 'requirements.txt' to import all associated dependancies

Model Analysis

My final model metrics are in the 'eval_metrics.md' write up.