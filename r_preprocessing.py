from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, f1_score, r2_score,roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import catboost as cb
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from rapidfuzz import fuzz, process
import shap
import matplotlib.pyplot as plt
from scipy.stats import linregress

class Preprocessing():
    def __init__():


        def read_merge_external(income_data,zipcode):
            dataset = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/properties_data_cleaned_05_12_14H30.csv"
            income_data = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/INCOME DATA 2022.csv"
            zipcode = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/BELGIUM/zipcodes_num_nl_new_Tumi.xls"
            id = pd.read_csv(income_data)
            
            zcode = pd.read_excel(zipcode)
    
            #Clean Income Data
            id_new_header = ["Locality", "min_median_income", "unnamed", "max_median_income", "locality"]
            id.columns = id_new_header
            id = id.drop(columns=["Locality", "unnamed"])

            #rename columns from bpPost
            zcode = zcode.rename(columns={'Localité': 'locality'})
            #Change bpost postal code column names
            zcode.rename(columns={'Postcode': 'postal_code', 'Provincie': 'province','NAME':'locality','MAIN MUNICIPALITY':"municipality"}, inplace=True)
            #Create blank columns for income data CSV
            id["postal_code"] = None
            id["province"] = None

            #Format data types between bpost dataset and Income data
            zcode["province"] = zcode["province"].astype(str)
            zcode["locality"] = zcode["locality"].astype(str)
            zcode["municipality"] = zcode["municipality"].astype(str)
            zcode["province"] = zcode["province"].apply(lambda x: x.strip().lower())
            zcode["locality"] = zcode["locality"].apply(lambda x: x.strip().lower())
            zcode["municipality"] = zcode["municipality"].apply(lambda x: x.strip().lower())
            id["locality"] = id["locality"].apply(lambda x: x.strip().lower())
            id["locality"] = id["locality"].astype(str)

            #Change Income data column names
            id.rename(columns={'max_median_income': 'gps_coordinates', "min_median_income" : "median_income"}, inplace=True)
            #Drop GPS Coordinates
            id = id.drop(columns="gps_coordinates")

            #Merge postal code & provice data from Bpost to income data
            for index, row in id.iterrows():
            # Normalize the 'locality' column in both DataFrames
                id_locality = row['locality'].strip().lower() if isinstance(row['locality'], str) else ''
                zcode['normalized_locality'] = zcode['locality'].apply(lambda x: x.strip().lower() if isinstance(x, str) else '')
                
                # Find matching rows based on normalized 'locality'
                matching_row = zcode[zcode['normalized_locality'] == id_locality]
         
                if not matching_row.empty:
                    # Copy relevant data from matching_row to id
                    id.at[index, 'postal_code'] = matching_row['postal_code'].values[0]
                    id.at[index, 'province'] = matching_row['province'].values[0]
                    id.at[index, 'new_column'] = 'Match Found'
                else:
                    # Add a flag if no match is found
                    id.at[index, 'new_column'] = 'No Match'
        
            return id, zcode

        def match_income_with_dataset(row, id, log_file="match_results.csv", unmatched_file="unmatched_results. csv"):
            """
            Match income based on postal code or locality.
            
            Priority:
            1. Exact match on postal code.
            2. Fuzzy match on locality.
            
            If no match is found, return None.
            Log each iteration to a CSV for double-checking.
            """
            # Create a dictionary to log results for this row
            log_data = {
                "postal_code": row['postal_code'],
                "locality": row['locality'],
                "province": row['province'],
                "match_type": "None",
                "matched_value": None,
                "median_income": None
            }

            # Attempt exact match on postal code
            postal_matches = id[id['postal_code'] == row['postal_code']]
            if not postal_matches.empty:
                matched_income = postal_matches['median_income'].values[0]
                log_data.update({
                    "match_type": "Postal Code",
                    "matched_value": row['postal_code'],
                    "median_income": matched_income
                })
                append_to_log(log_data, log_file)
                return matched_income

            # Fuzzy match on locality
            best_match = process.extractOne(row['locality'], id['locality'])
            if best_match and best_match[1] > 75:  # Ensure the match score is above a threshold
                matched_locality = best_match[0]
                matched_income = id[id['locality'] == matched_locality]['median_income'].values[0]
                log_data.update({
                    "match_type": "Locality",
                    "matched_value": matched_locality,
                    "median_income": matched_income
                })
                append_to_log(log_data, log_file)
                return matched_income

            # No match found; log to the unmatched file
            append_to_log(log_data, unmatched_file)
            return None

        def append_to_log(log_data, log_file):
            """
            Append log data to the specified CSV file.
            """
            # Convert the log data dictionary to a DataFrame
            log_df = pd.DataFrame([log_data])

            # Append to the CSV file
            try:
                # If the file exists, append without writing the header
                log_df.to_csv(log_file, mode='a', index=False, header=False)
            except FileNotFoundError:
                # If the file does not exist, write with the header
                log_df.to_csv(log_file, mode='w', index=False, header=True)


        def properties_dataset_cleaning(property_filepath,match_income_with_dataset):
            
            df = pd.read_csv(property_filepath)
            #Additional data cleaning and renaming
            df = df[df['price'] <= 5000000]
            df = df[df['price'] >= 40000]
            df = df[df['bedrooms'] <= 9]

            #Encode Building state
            df['buildingState'] = df['buildingState'].replace({
                'AS_NEW': 1,
                'JUST_RENOVATED': 2,
                'GOOD': 3,
                'TO_RESTORE': 4,
                'TO_RENOVATE': 4,
                'TO_BE_DONE_UP':4
            })

            df['province'] = df['province'].replace({'flemish_brabant_extended': 'flemish_brabant', 'hainaut_extended': 'hainaut_province'})
            df['province'] = df['province'].replace({'flemish_brabant': 'Flemish Brabant', 'hainaut_province': 'Hainaut', 'antwerp_province': 'Antwerp', 'brussels_capital': 'Brussels', 
                                                    'limburg_province': 'Limburg', 'liège_province': 'Liège', 'luxembourg_province': 'Luxembourg', 'namur_province': 'Namur', 
                                                    'walloon_brabant': 'Walloon Brabant', 'west_flanders': 'West Flanders','east_flanders': 'East Flanders'})

            #Cleaning
            columns_to_replace = ['terraceSurface']
            df[columns_to_replace] = df[columns_to_replace].fillna(0)
            df = df.dropna(subset=['livingArea',"energy_certificate"])
            df = df.drop(['buildingStateLabel'], axis=1)        
            df2=df.copy()
            df2["median_income"] = None
            df2["postal_code"].astype(int)
            df2["locality"] = df2["locality"].str.lower().str.strip()
            #Apply FuzzyMatch
            df2['median_income'] = df2.apply(match_income_with_dataset, axis=1, id=id)
            #Clear missing values from median income
            missing_media = df2[df2["median_income"].isna()]
            df3 = df2.drop(missing_media.index)
            
            #Scale income price values
            #Convert number formatting of median_income column
            df3['median_income'] = df3['median_income'] * 1000
            #make a float
            df3['median_income'] = df3['median_income'].astype(float)
            return df3
        
        
        
class Feature_Engineering():
    def __init__():



        def data_preparation(dataframe):
            df = dataframe
            #Format dtype
            df["bedrooms"]= df["bedrooms"].astype(int)
            df["postal_code"]=df["postal_code"].astype(int)

        # Define a function to assign regions to provinces
        def assign_region(df,province):
            flanders_provinces = ["Antwerp", "East Flanders", "Flemish Brabant", "Limburg", "West Flanders"]
            wallonia_provinces = ["Liège", "Luxembourg", "Walloon Brabant", "Namur", "Hainaut"]
            if province in flanders_provinces:
                return "Flanders"
            elif province in wallonia_provinces:
                return "Wallonia"
            else:
                return "Brussels"
 
        def inter_quantile(df):
            #Remove Inter-quantile Range and adding 1.5times IQR for outliers
            q1 = df['price'].quantile(0.25)
            q3 = df['price'].quantile(0.75)
            iqr = q3 - q1
            df = df[(df['price'] >= q1 - 1.5 * iqr) & (df['price'] <= q3 + 1.5 * iqr)]


class Model_Apply():
    def __init__():


        def instantiate_model(df):
            #define X and Y for model
            X = df.drop(columns=["price","kitchen","postal_code","furnished","fireplace","province","property_type","terraceSurface"])
            y = df["price"] 
            #Train Test Split
            X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
            #Categorical Features for Catboost
            categorical_feature_indices = ["locality","energy_certificate","region"]
            #KFold Splits instantiate
            kf = KFold(n_splits=5,shuffle=True, random_state=42)
            feature_importances = []
            #BEST PARAMETERS LINEAR 95th Percentile
            best_params = {
                "random_strength": 5,
                "learning_rate": 0.07444444444444444,
                "l2_leaf_reg": 7,
                "iterations": 1500,
                "depth": 6,
                "eval_metric": "RMSE",
                "verbose": 100,  # Adjust verbosity to see training progress
                "random_seed": 42,
                "cat_features": categorical_feature_indices
            }

            # Initialize the CatBoost model
            model = CatBoostRegressor(**best_params)

            # Train the model
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

            # Make predictions
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # Append feature importance
            feature_importances.append(model.get_feature_importance())

            return categorical_feature_indices,X_train, y_train,X_test, y_test,y_pred,y_train_pred,y_test_pred,feature_importances

class Model_evaluation():

    def shap_analysis(categorical_feature_indices,X_train, y_train,X_test, y_test,y_pred,y_train_pred,y_test_pred,feature_importances):
        # Cat Features for Shap
        categorical_feature_indices = [X_test.columns.get_loc(col) for col in ["locality","energy_certificate","region"]]

        # Create Pool with explicit categorical feature indices
        test_pool = Pool(
            X_test, 
            y_test, 
            cat_features=categorical_feature_indices
        )

        # SHAP analysis
        print("\nPerforming SHAP analysis...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Generate SHAP summary plot
        print("Generating SHAP summary plot...")
        summary_plot = shap.summary_plot(shap_values, X_test)

        # SHAP dependence plot for a specific feature
        specific_feature = "price"  # Replace with your feature of interest
        if specific_feature in X_test.columns:
            print(f"Generating SHAP dependence plot for {specific_feature}...")
            dependance_plot = shap.dependence_plot(specific_feature, shap_values, X_test)

        
        return explainer, summary_plot, dependance_plot
    
    def top_features():
        # Average feature importance across all folds
        avg_importance = np.mean(feature_importances, axis=0)
        
        # Convert to DataFrame
        feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
        
        # Sort and take the top 20 features
        top_features = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)
        
        # Set the style and color palette
        sns.set_style("whitegrid")
        palette = sns.color_palette("rocket", len(top_features))
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        ax = sns.barplot(x='Importance', y='Feature', data=top_features, palette=palette)
        
        # Customize the plot
        plt.title('Top 20 Most Important Features - CatBoost Model', fontsize=20, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=15)
        plt.ylabel('Features', fontsize=15)
        
        # Add value labels to the end of each bar
        for i, v in enumerate(top_features['Importance']):
            ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=13)
        
        # Extend x-axis by 10% and feature names font size
        plt.xlim(0, max(top_features['Importance']) * 1.1)
        plt.yticks(fontsize=13)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()


    def linear_regression_plot(y_test,y_pred):
        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color="skyblue", label="Predicted vs Actual")

        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
        regression_line = slope * np.array(y_test) + intercept

        # Plot regression line
        plt.plot(y_test, regression_line, color="red", label=f"Regression Line (R² = {r_value**2:.2f})")

        #
        #labels, title, and legend
        plt.title("Actual vs Predicted with Regression Line")
        plt.xlabel("Actual Values (y_test)")
        plt.ylabel("Predicted Values (y_pred)")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.show()
    
    def test_vs_training(y_test,y_pred,y_train,y_train_pred):
       # Calculate metrics
        mse = mean_squared_error(y_test,y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
        mae_test = metrics.mean_absolute_error(y_test, y_pred)
        msle = metrics.mean_squared_log_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        #Compare Training & Test data
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        #R2 train & test comparission
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_pred)

        #Print Scores
        print(f"Test RMSE: {rmse}, Test R2: {r2}, Test MSE: {mse}, TEST MAE: {mae_test}, Test MSLE: {msle}")
        print(f"Training MAE: {mae_train}, Test RMSE: {mae_test}")
        print(f"Training RMSE: {train_rmse}, Test RMSE: {test_rmse}")
        print(f"Training R²: {train_r2}, Test R²: {test_r2}") 

    def training_vs_validation_loss():
        # Extract loss values
        loss = model.evals_result_
        train_loss = loss['learn']['RMSE']
        test_loss = loss['validation']['RMSE']

        # Plot training vs validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label="Training Loss", color="blue")
        plt.plot(test_loss, label="Validation Loss", color="orange")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()



def main():
    dataset = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/properties_data_cleaned_05_12_14H30.csv"
    income_data = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/INCOME DATA 2022.csv"
    zipcode = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/BELGIUM/zipcodes_num_nl_new_Tumi.xls"
    external_data, property_dataset  = Preprocessing() 
    external_data.read_merge_external(income_data,zipcode)
    property_dataset.properties_dataset_cleaning()
    
    df = pd
    fe = Feature_Engineering()
    #Feature engineering
    df["region"] = df["province"].apply(assign_region)