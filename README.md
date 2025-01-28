# Regression Project Documentation

## Project Overview
This project focuses on real estate price prediction using machine learning techniques. The pipeline is structured into four key stages:

### 1. Preprocessing
- Cleans and merges datasets related to properties, income, and postal codes.
- Utilizes external data sources:
  - **Property Data**: `properties_data_cleaned_05_12_14H30`
  - **Income Data**: `INCOME DATA 2022` (sourced from [Statbel](https://statbel.fgov.be/en/news/attert-richest-municipality-and-saint-josse-ten-noode-poorest-2022))
  - **Zip Code Data**: `zipcodes_num_nl_new_Tumi` (sourced from [BPpost](https://www.bpost.be/fr/outil-de-validation-de-codes-postaux))
- Uses **rapidfuzz** for approximate string matching between French and Dutch names, achieving 75% accuracy.

### 2. Feature Engineering
- Cleans dataset by removing outliers and handling missing values.
- Creates a **regional categorization** for properties in **Brussels**, **Wallonia**, and **Flanders**.
- Drops features with low correlation to price, including:
  - `kitchen`, `postal_code`, `furnished`, `fireplace`, `province`, `property_type`, `Terrace Surface`
- Scales the median income data to align with property prices.
- Removes extreme outliers based on **1.5× Interquartile Range (IQR)** while retaining some variation.

### 3. Model Training and Application
- Uses **CatBoost** for model training with cross-validation.
- Performs **random search** for hyperparameter tuning.
- Incorporates categorical features: `locality`, `energy_certificate`, `region`.

### 4. Model Evaluation & Interpretation
- Evaluates performance using multiple metrics:
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**
  - **R² Score (Coefficient of Determination)**
  - **MAPE (Mean Absolute Percentage Error)**
  - **sMAPE (Symmetric Mean Absolute Percentage Error)**
- Visualizes results and performs **SHAP (SHapley Additive exPlanations)** analysis for feature importance.

---

## Model Performance

### Best Model Metrics
- **Total code runtime**: 179.493 seconds
- **Model training time**: 25.66 seconds

| Metric       | Train  | Test  |
|-------------|--------|-------|
| **MAE**     | 53,046.11  | 62,511.11  |
| **RMSE**    | 72,935.64  | 86,764.82  |
| **R² Score**| 0.8018  | 0.7318  |
| **MAPE**    | 16.10%  | 18.79%  |
| **sMAPE**   | 15.23%  | 17.67%  |

### Worst Model Metrics
| Metric       | Train  | Test  |
|-------------|--------|-------|
| **MAE**     | 38,377.06  | 45,291.87  |
| **RMSE**    | 47,976.58  | 56,634.58  |
| **R² Score**| 0.5481  | 0.4130  |
| **MAPE**    | 11.08%  | 12.88%  |
| **sMAPE**   | 10.91%  | 12.69%  |

---

## Insights from Model Evaluation

### 1. Median Income Impact
- Median income per municipality had **limited correlation** with property prices.
- The metric was relevant for middle-income housing but less so for **luxury or low-cost properties**.

### 2. Energy Certification Impact
- The energy certification **significantly influenced prices in Flanders**, where regulatory compliance is strict.
- This effect was weaker in **Brussels and Wallonia**, possibly due to different taxation policies.

---

## Prerequisites & Dependencies

### Software Requirements
- Python 3.7+

### Libraries Used
- `pandas`
- `numpy`
- `sklearn`
- `seaborn`
- `matplotlib`
- `CatBoost`
- `shap`
- `rapidfuzz`
- `scipy`

*Refer to* `requirements.txt` *for a complete list of dependencies.*

---

## Next Steps
- **Refine feature selection** based on further SHAP analysis.
- **Test additional regression models** (e.g., XGBoost, LightGBM) for comparison.
- **Investigate regional disparities** in energy certification policies for better model accuracy.
