# regression
Immo Regression
Project Structure

The project is divided into four main classes:

Preprocessing: Cleans and merges income, zipcode, and property datasets.
    # Data in utils
    property_data CSV = "properties_data_cleaned_05_12_14H30"
    income_data CSV = "INCOME DATA 2022"
    zipcode_data from Bpost = "zipcodes_num_nl_new_Tumi"

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
    "VenV_immo" is included in the utils

Model Analysis

My Best model metrics are in the 'eval_metrics.md' write up with a 
total code running time: 179.493 seconds
Model training time: 25.66 seconds.
# Noted in evaluation_metrics folder

MAE_train = 53046.11454390735
MAE_test = 62511.11393412253
RMSE_train = 72935.63958945835
RMSE_test = 86764.82096128115
R2_train = 0.8018401136632958
R2_test = 0.7318702663432397
MAPE_train = 16.10101937010805
MAPE_test = 18.794841133668257
sMAPE_train = 15.228928430096653
sMAPE_test = 17.66769666366581
								

My worst metrics
# Noted in the worst evaulation metrics

MAE_train = 38377.05823428117
MAE_test = 45291.87155948026
RMSE_train = 47976.581473187165
RMSE_test = 56634.576269188365
R2_train = 0.5480939828309246
R2_test = 0.4130440897552218
MAPE_train = 11.084191483776573
MAPE_test = 12.882497602247009
sMAPE_train = 10.908931576969566
sMAPE_test = 12.6929203922517
