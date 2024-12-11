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
import time


class Preprocessing:
    def __init__(self, income_path, zipcode_path, property_path):
        self.income_path = income_path
        self.zipcode_path = zipcode_path
        self.property_path = property_path
        """
        Preprocessing class:
        This is where I match the Bpost location data and municipalities with the Statbel
        Median income information.        

        """
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
        #Clear outliers above 5 million
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

        # Functions for MAPE and sMAPE
        def mape(y_true, y_pred):
            """Mean Absolute Percentage Error"""
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        def smape(y_true, y_pred):
            """Symmetric Mean Absolute Percentage Error"""
            return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        
        # Metrics
        metrics = {
            "MAE_train": mean_absolute_error(y_train, y_train_pred),
            "MAE_test": mean_absolute_error(y_test, y_test_pred),
            "RMSE_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "RMSE_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "R2_train": r2_score(y_train, y_train_pred),
            "R2_test": r2_score(y_test, y_test_pred),
            "MAPE_train": mape(y_train, y_train_pred),
            "MAPE_test": mape(y_test, y_test_pred),
            "sMAPE_train": smape(y_train, y_train_pred),
            "sMAPE_test": smape(y_test, y_test_pred),
        }

        return metrics

    @staticmethod
    def shap_analysis(model, X_test, y_test, categorical_features):
        """
        Performs SHAP analysis on the model to understand feature contributions and feature interactions.

        Includes both gain-based and mean split importance analyses.

        Outputs:
            SHAP summary plot, dependence plots, and interaction plots (both gain-based and mean split) are saved as images.
        """
        # Create a test Pool with categorical features
        test_pool = Pool(X_test, y_test, cat_features=categorical_features)

        # Initialize SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_pool)

        # Create directory to store SHAP outputs
        shap_output_dir = "shap_outputs"
        os.makedirs(shap_output_dir, exist_ok=True)

        # Plotting to prevent cutoff
        plt.rcParams.update({
            'figure.autolayout': True,
            'figure.figsize': (12, 8),  # Larger figure size
            'figure.constrained_layout.use': True,
        })

        # SHAP Summary Plot (global feature importance, mean split)
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_test, show=False, plot_type='bar')
        plt.title("SHAP Summary Plot (Mean Split)")
        plt.tight_layout(pad=3.0)  # Add extra padding
        plt.savefig(f"{shap_output_dir}/shap_summary_plot_mean.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"SHAP Summary Plot (mean split) saved to {shap_output_dir}/shap_summary_plot_mean.png")

        # Gain-Based Feature Importances
        gain_importances = model.get_feature_importance(type='PredictionValuesChange')
        gain_importance_indices = np.argsort(gain_importances)[::-1]  # Sort in descending order
        gain_top_features = [X_test.columns[i] for i in gain_importance_indices[:5]]  # Top 5 features

        # SHAP Summary Plot for Gain-Based Importance
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
        plt.title("SHAP Summary Plot (Gain-Based)")
        plt.tight_layout(pad=3.0)
        plt.savefig(f"{shap_output_dir}/shap_summary_plot_gain.png", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"SHAP Summary Plot (gain-based split) saved to {shap_output_dir}/shap_summary_plot_gain.png")

        # Generate SHAP Dependence Plots (Mean Split)
        for feature in X_test.columns:
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(feature, shap_values, X_test, show=False)
            plt.title(f"SHAP Dependence Plot: {feature} (Mean Split)")
            plt.tight_layout(pad=3.0)
            plt.savefig(f"{shap_output_dir}/shap_dependence_mean_{feature}.png", bbox_inches='tight', dpi=300)
            plt.close()
            print(f"SHAP Dependence Plot (mean split) for {feature} saved to {shap_output_dir}/shap_dependence_mean_{feature}.png")

        # Generate SHAP Dependence Plots (Gain-Based)
        for feature in gain_top_features:
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(feature, shap_values, X_test, show=False)
            plt.title(f"SHAP Dependence Plot: {feature} (Gain-Based)")
            plt.tight_layout(pad=3.0)
            plt.savefig(f"{shap_output_dir}/shap_dependence_gain_{feature}.png", bbox_inches='tight', dpi=300)
            plt.close()
            print(f"SHAP Dependence Plot (gain-based split) for {feature} saved to {shap_output_dir}/shap_dependence_gain_{feature}.png")

        # Generate SHAP Interaction Plots (Mean Split)
        mean_split_top_features_indices = np.argsort(np.abs(shap_values).mean(0))[-5:]  # Top 5 features by mean split
        mean_split_top_features = [X_test.columns[i] for i in mean_split_top_features_indices]

        for i, feature_x in enumerate(mean_split_top_features):
            for j, feature_y in enumerate(mean_split_top_features):
                if i != j:  # Avoid self-interactions
                    plt.figure(figsize=(12, 8))
                    shap.dependence_plot(feature_x, shap_values, X_test, show=False, interaction_index=feature_y)
                    plt.title(f"SHAP Interaction Plot: {feature_x} vs {feature_y} (Mean Split)")
                    plt.tight_layout(pad=3.0)
                    plt.savefig(f"{shap_output_dir}/shap_interaction_mean_{feature_x}_vs_{feature_y}.png", bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"SHAP Interaction Plot (mean split) for {feature_x} vs {feature_y} saved to {shap_output_dir}/shap_interaction_mean_{feature_x}_vs_{feature_y}.png")

        # Generate SHAP Interaction Plots (Gain-Based)
        for i, feature_x in enumerate(gain_top_features):
            for j, feature_y in enumerate(gain_top_features):
                if i != j:  # Avoid self-interactions
                    plt.figure(figsize=(12, 8))
                    shap.dependence_plot(feature_x, shap_values, X_test, show=False, interaction_index=feature_y)
                    plt.title(f"SHAP Interaction Plot: {feature_x} vs {feature_y} (Gain-Based)")
                    plt.tight_layout(pad=3.0)
                    plt.savefig(f"{shap_output_dir}/shap_interaction_gain_{feature_x}_vs_{feature_y}.png", bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"SHAP Interaction Plot (gain-based split) for {feature_x} vs {feature_y} saved to {shap_output_dir}/shap_interaction_gain_{feature_x}_vs_{feature_y}.png")




    
    
    @staticmethod
    def plot_training_validation_loss(model, output_dir):
        """
        Plot training and validation loss across iterations
        
        Args:
            model (CatBoostRegressor): Trained CatBoost model
            output_dir (str): Directory to save the plot
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract learning curves
        train_loss = model.get_evals_result()['learn']['RMSE']
        validation_loss = model.get_evals_result()['validation']['RMSE']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss (RMSE)', color='blue')
        plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss (RMSE)', color='red')
        
        plt.title('Training vs Validation Loss')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'worst_training_validation_loss.png'))
        plt.close()

    @staticmethod
    def plot_prediction_scatter(y_test, y_pred, output_dir):
        """
        Create a scatter plot of actual vs predicted values with linear regression line
        
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
        
        # Compute linear regression
        slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
        line = slope * y_test + intercept
        
        # Plot regression line
        plt.plot(y_test, line, color='red', label=f'Regression Line (R²: {r_value**2:.4f})')
        
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'worst_actual_vs_predicted_scatter.png'))
        plt.close()

    @staticmethod
    def export_evaluation_results(metrics, feature_importances, X_train, file_path, model=None, y_test=None, y_pred=None, export_type="csv"):
        """
        Extended export method to include additional visualizations
        
    
        """
        # Create evaluation metrics directory
        output_dir = os.path.dirname(file_path)
        evaluation_metrics_dir = os.path.join(output_dir, 'evaluation_metrics')
        os.makedirs(evaluation_metrics_dir, exist_ok=True)

        # Create SHAP outputs directory
        shap_output_dir = os.path.join(evaluation_metrics_dir, 'shap_outputs')
        os.makedirs(shap_output_dir, exist_ok=True)

        # Save Metrics
        metrics_file_path = os.path.join(evaluation_metrics_dir, 'evaluation_results.csv')
        if export_type == "csv":
            pd.DataFrame([metrics]).to_csv(metrics_file_path, index=False)
        elif export_type in ("txt", "md"):
            with open(metrics_file_path, "w") as f:
                f.write("# Model Evaluation Results\n\n" if export_type == "md" else "")
                for key, value in metrics.items():
                    f.write(f"- **{key}**: {value}\n" if export_type == "md" else f"{key}: {value}\n")

        # Dynamically retrieve feature names
        feature_names = X_train.columns.tolist()

        # Aggregate feature importances by taking the mean across folds
        mean_feature_importances = (
            np.mean(feature_importances, axis=0) if isinstance(feature_importances, list) else feature_importances
        )

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": mean_feature_importances}
        ).sort_values(by="Importance", ascending=False)

        # Save Feature Importances Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis")
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_metrics_dir, 'feature_importances.png'))
        plt.close()

        # Additional Visualizations (if model and predictions are provided)
        if model is not None:
            # Plot Training vs Validation Loss
            ModelEvaluation.plot_training_validation_loss(model, evaluation_metrics_dir)
        
        if y_test is not None and y_pred is not None:
            # Plot Actual vs Predicted Scatter
            ModelEvaluation.plot_prediction_scatter(y_test, y_pred, evaluation_metrics_dir)

        print(f"Evaluation metrics and visualizations saved to {evaluation_metrics_dir}")


# Main Function
def main():
    # Paths
    script_start_time = time.time()
    print("Script started.")
    income_path = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/Delivery/INCOME DATA 2022.csv"
    zipcode_path = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/Delivery/zipcodes_num_nl_new_Tumi.xls"
    property_path = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/Delivery/properties_data_cleaned_05_12_14H30.csv"

    # Preprocessing
    preprocessing = Preprocessing(income_path, zipcode_path, property_path)
    income_data, zipcode_data = preprocessing.read_merge_external()
    property_data = preprocessing.properties_dataset_cleaning()

    # Feature Engineering
    feature_engineering = FeatureEngineering()
    property_data = feature_engineering.add_region_column(property_data)
    property_data = feature_engineering.remove_outliers_iqr(property_data)

    # Model Training
    training_start_time = time.time()
    print("Model training started...")
    model_apply = ModelApply()
    model, X_train, X_test, y_train, y_test, feature_importances = model_apply.train_model(property_data)
    
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    print(f"Model training completed in {training_duration:.2f} seconds.")
    # Model Evaluation
    evaluation = ModelEvaluation()
    metrics = ModelEvaluation.evaluate_model(model, X_train, y_train, X_test, y_test)
    # SHAP Analysis
    ModelEvaluation.shap_analysis(model, X_test, y_test, ["locality", "energy_certificate", "region"])
    
    #Gain based importance
    #gain_importances = evaluation.calculate_gain_importance(model)
    # y_test_pred for scatter plot and loss curves 
    y_test_pred = model.predict(X_test)

    # Export Results
    evaluation_start_time = time.time()
    print("Evaluation started...")
    ModelEvaluation.export_evaluation_results(
    metrics, 
    feature_importances, 
    X_train, 
    "evaluation_results.csv", 
    model=model,  # Pass the model for loss curves
    y_test=y_test,  # Pass actual test values
    y_pred=y_test_pred,  # Pass predicted test values
    export_type="csv"
    
)   
    time.sleep(2)  
    evaluation_end_time = time.time()
    # Evaluation duration
    evaluation_duration = evaluation_end_time - evaluation_start_time
    print(f"Evaluation completed in {evaluation_duration:.2f} seconds.")

    # Capture the script end time
    script_end_time = time.time()

    # Total script execution time
    script_execution_time = script_end_time - script_start_time
    print(f"Total script execution time: {script_execution_time:.2f} seconds.")


if __name__ == "__main__":
    main()