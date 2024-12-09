#Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from catboost import CatBoostRegressor, Pool
from rapidfuzz import process
import shap
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import os


class Preprocessing:
    def __init__(self, income_path, zipcode_path, property_path):
        self.income_path = income_path
        self.zipcode_path = zipcode_path
        self.property_path = property_path

    def read_merge_external(self):
        # Read income and zipcode data
        income_data = pd.read_csv(self.income_path)
        zipcode_data = pd.read_excel(self.zipcode_path)

        # DEBUG: Print the columns of the input dataframes
        print("Income Data Columns:", income_data.columns)
        print("Zipcode Data Columns:", zipcode_data.columns)
        #Clean Income Data
        income_data_new_header = ["Locality", "min_median_income", "unnamed", "max_median_income", "locality"]
        income_data.columns = income_data_new_header
        income_data = income_data.drop(columns=["Locality", "unnamed"])

        # Ensure all column names are lowercase for consistent renaming
        zipcode_data.columns = zipcode_data.columns.str.lower()

        # Rename columns in the zipcode_data dataframe
        if "main municipality" in zipcode_data.columns:
            zipcode_data.rename(
                columns={"postcode": "postal_code", "provincie": "province", "name": "locality", "main municipality": "municipality"},
                inplace=True,
            )
        else:
            raise KeyError("The column 'MAIN MUNICIPALITY' is missing from the input file. Please check the file structure.")

        # Normalize text for merging
        zipcode_data.province = zipcode_data["province"].astype(str)
        zipcode_data.locality = zipcode_data["locality"].astype(str)
        zipcode_data.municipality = zipcode_data["municipality"].astype(str)
        zipcode_data.province = zipcode_data.province.apply(lambda x: x.strip().lower())
        zipcode_data.locality = zipcode_data.locality.apply(lambda x: x.strip().lower())
        zipcode_data.municipality = zipcode_data.municipality.apply(lambda x: x.strip().lower())
        income_data.locality = income_data.locality.apply(lambda x: x.strip().lower())
        income_data.locality = income_data["locality"].astype(str)

        # Merge postal code & province data from Bpost to income data
        for index, row in income_data.iterrows():
            id_locality = row["locality"]
            match = zipcode_data[zipcode_data["locality"] == id_locality]
            if not match.empty:
                income_data.at[index, "postal_code"] = match["postal_code"].values[0]
                income_data.at[index, "province"] = match["province"].values[0]

        # RapidFuzz matching: For unmatched rows, use fuzzy matching
        for index, row in income_data.iterrows():
            if pd.isnull(row["postal_code"]) or pd.isnull(row["province"]):
                locality = row["locality"]
                match = process.extractOne(locality, zipcode_data["locality"], score_cutoff=75)
                if match:
                    match_row = zipcode_data[zipcode_data["locality"] == match[0]]
                    income_data.at[index, "postal_code"] = match_row["postal_code"].values[0]
                    income_data.at[index, "province"] = match_row["province"].values[0]

        return income_data, zipcode_data


    def properties_dataset_cleaning(self):
        # Read and clean property data
        property_data = pd.read_csv(self.property_path)
        property_data = property_data[
            (property_data["price"] <= 5000000) & (property_data["price"] >= 40000)
        ]
        property_data = property_data[property_data["bedrooms"] <= 9]
        property_data["buildingState"] = property_data["buildingState"].replace(
            {
                "AS_NEW": 1,
                "JUST_RENOVATED": 2,
                "GOOD": 3,
                "TO_RESTORE": 4,
                "TO_RENOVATE": 4,
                "TO_BE_DONE_UP": 4,
            }
        )
        property_data["province"] = property_data["province"].replace(
            {
                "flemish_brabant_extended": "flemish_brabant",
                "hainaut_extended": "hainaut_province",
                "flemish_brabant": "Flemish Brabant",
                "hainaut_province": "Hainaut",
            }
        )
        property_data.drop(columns=["buildingStateLabel"], inplace=True)

        # Fill and clean missing values
        property_data["terraceSurface"] = property_data["terraceSurface"].fillna(0)
        property_data.dropna(subset=["livingArea", "energy_certificate"], inplace=True)
        return property_data


class FeatureEngineering:
    @staticmethod
    def add_region_column(df):
        flanders_provinces = [
            "Antwerp",
            "East Flanders",
            "Flemish Brabant",
            "Limburg",
            "West Flanders",
        ]
        wallonia_provinces = [
            "Liège",
            "Luxembourg",
            "Walloon Brabant",
            "Namur",
            "Hainaut",
        ]
        df["region"] = df["province"].apply(
            lambda province: "Flanders"
            if province in flanders_provinces
            else "Wallonia"
            if province in wallonia_provinces
            else "Brussels"
        )
        return df

    @staticmethod
    def remove_outliers_iqr(df):
        # Interquartile Range (IQR) Method
        q1 = df["price"].quantile(0.25)
        q3 = df["price"].quantile(0.75)
        iqr = q3 - q1
        df = df[(df["price"] >= q1 - 1.5 * iqr) & (df["price"] <= q3 + 1.5 * iqr)]
        return df


class ModelApply:
    @staticmethod
    def train_model(df):
        # Prepare data for training
        X = df.drop(
            columns=[
                "price",
                "kitchen",
                "postal_code",
                "furnished",
                "fireplace",
                "province",
                "property_type",
                "terraceSurface",
            ]
        )
        y = df["price"]

        # Categorical features
        categorical_features = ["locality", "energy_certificate", "region"]

        # KFold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        feature_importances = []

        params = {
            "random_strength": 5,
            "learning_rate": 0.07,
            "l2_leaf_reg": 7,
            "iterations": 1500,
            "depth": 6,
            "eval_metric": "RMSE",
            "verbose": 100,
            "random_seed": 42,
            "cat_features": categorical_features,
        }

        # Train with KFold
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
            feature_importances.append(model.get_feature_importance())

        return model, X_train, X_test, y_train, y_test, feature_importances


class ModelEvaluation:
    @staticmethod
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "MAE_train": mean_absolute_error(y_train, y_train_pred),
            "MAE_test": mean_absolute_error(y_test, y_test_pred),
            "RMSE_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "RMSE_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "R2_train": r2_score(y_train, y_train_pred),
            "R2_test": r2_score(y_test, y_test_pred),
        }

        return metrics

    @staticmethod
    def shap_analysis(model, X_test, y_test, categorical_features):
        # SHAP Analysis with Test Pool
        test_pool = Pool(X_test, y_test, cat_features=categorical_features)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_pool)

        # SHAP Summary Plot
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("shap_summary_plot.png")

        # SHAP Dependence Plot for one feature (e.g., "price")
        if "price" in X_test.columns:
            shap.dependence_plot("price", shap_values, X_test, show=False)
            plt.savefig("shap_dependence_plot.png")

    @staticmethod
    def export_evaluation_results(metrics, feature_importances, X_train, file_path, export_type="csv"):
        # Save Metrics
        if export_type == "csv":
            pd.DataFrame([metrics]).to_csv(file_path, index=False)
        elif export_type in ("txt", "md"):
            with open(file_path, "w") as f:
                f.write("# Model Evaluation Results\n\n" if export_type == "md" else "")
                for key, value in metrics.items():
                    f.write(f"- **{key}**: {value}\n" if export_type == "md" else f"{key}: {value}\n")

        # Dynamically retrieve feature names
        feature_names = X_train.columns.tolist()

        # Aggregate feature importances by taking the mean across folds
        if isinstance(feature_importances, list):
            # Convert list of numpy arrays to numpy array
            feature_importances_array = np.array(feature_importances)
            
            # Calculate mean importance for each feature across all folds
            mean_feature_importances = np.mean(feature_importances_array, axis=0)
        else:
            mean_feature_importances = feature_importances

        # Check if lengths match
        if len(mean_feature_importances) != len(feature_names):
            raise ValueError(
                f"Length mismatch: feature_importances ({len(mean_feature_importances)}) vs feature_names ({len(feature_names)})"
            )

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": mean_feature_importances}
        ).sort_values(by="Importance", ascending=False)

        # Plot Feature Importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        plt.close()



# Main Function
def main():
    # Paths
    income_path = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/INCOME DATA 2022.csv"
    zipcode_path = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/BELGIUM/zipcodes_num_nl_new_Tumi.xls"
    property_path = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/properties_data_cleaned_05_12_14H30.csv"

    # Preprocessing
    preprocessing = Preprocessing(income_path, zipcode_path, property_path)
    income_data, zipcode_data = preprocessing.read_merge_external()
    property_data = preprocessing.properties_dataset_cleaning()

    # Feature Engineering
    feature_engineering = FeatureEngineering()
    property_data = feature_engineering.add_region_column(property_data)
    property_data = feature_engineering.remove_outliers_iqr(property_data)

    # Model Training
    model_apply = ModelApply()
    model, X_train, X_test, y_train, y_test, feature_importances = model_apply.train_model(property_data)

    # Model Evaluation
    evaluation = ModelEvaluation()
    metrics = evaluation.evaluate_model(model, X_train, y_train, X_test, y_test)

    # SHAP Analysis
    evaluation.shap_analysis(model, X_test, y_test, ["locality", "energy_certificate", "region"])

    # Export Results
    evaluation.export_evaluation_results(metrics, feature_importances, X_train, "evaluation_results.csv", export_type="csv")



if __name__ == "__main__":
    main()