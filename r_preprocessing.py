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
from fuzzywuzzy import process

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
            return df3
        
        
        








def main():
    dataset = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/properties_data_cleaned_05_12_14H30.csv"
    income_data = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/INCOME DATA 2022.csv"
    zipcode = "/Users/irisvirus/Desktop/Becode/Python/Projects/Regression/regression/BELGIUM/zipcodes_num_nl_new_Tumi.xls"
    external_data, property_dataset  = Preprocessing() 
    external_data.read_merge_external(income_data,zipcode)
    property_dataset.properties_dataset_cleaning()