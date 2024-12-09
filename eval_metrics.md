# %% [markdown]
# # Regression
# 
# 
# 
# 
# 
# 
# 

# %% [markdown]
# ## Preprocessing and Input Data
# 
# ### Downloaded External Information
# - I downloaded external open data from [Statbel](https://statbel.fgov.be/en/news/attert-richest-municipality-and-saint-josse-ten-noode-poorest-2022) on the median disposable income per municipality.
#   - This information accounted for both taxable and non-taxable (net) income, including:
#     - Professional income
#     - Social benefits
#     - Pensions
#     - Integration income
#     - Rental income
#     - Capital income
#     - Child allowances
#     - Maintenance allowances
#   - The dataset included the municipality and GPS coordinates.
# 
# - I downloaded postal codes per municipality from [BPpost](https://www.bpost.be/fr/outil-de-validation-de-codes-postaux).
# - I combined the income data and postal data to create a comprehensive list, enabling me to match the median income to my dataset using both postal code and municipality name.
# - To handle mismatches between French and Dutch names, I used the **rapidfuzz** library for approximate string matching, achieving 75% accuracy by matching on postal code and name.
# 
# ## Improving Property Dataset
# - I revisited the **Immoweb** site and scraped additional data to retrieve the energy certification scores for all previously scraped properties.
# - An article about a new law revealed that all properties would need to achieve level D compliance before 2030 to avoid higher taxes.

# %% [markdown]
# ## Feature Engineering
# - Cleaned the dataset by removing outliers and null values in critical features.
# - Created a new column to differentiate between **Brussels**, **Wallonia**, and **Flanders**, as public policies and politics vary significantly across these regions.
# - Dropped features with low correlation to price, including:
#   - `kitchen`, `postal_code`, `furnished`, `fireplace`, `province`, `property_type`, and `Terrace Surface`.
# - Scaled the median income data to align with the price scale.
# - Removed outliers beyond 1.5 times the interquartile range (IQR), retaining some for variability without introducing noise.
# 
# # Model Creation
# 
# - Used **Cross-Validation** and **Random Search** to optimize model performance.
# - Split the dataset into training and testing subsets to evaluate model accuracy.
# - I used the categorical features of ["locality", "energy_certificate", "region"]

# %% [markdown]
# # Interpretation of Results
# 
# ### Median Income Insights
# - Median income per municipality showed limited correlation with property prices.
# - The metric was more relevant for properties near the median price range, suggesting its importance primarily for middle-income households.
# - For extremely high or low-income groups, the metric held little significance.
# 
# ### Energy Certification Insights
# - The energy certification rating significantly impacted prices in the **Flanders** region, less so in **Wallonia** and **Brussels**.
# - This discrepancy is likely due to the 2030 compliance deadline enforced in Flanders, which does not apply to the other regions.
# - Analysis confirms a stronger relationship between energy certification and price in Flanders.


