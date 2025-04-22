import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Ensure NumPy is imported for any required operations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Step 1: Load the updated dataset
file_path = r"E:\Python project\Principal_Commodity_wise_export_for_the_year_202223.csv"  # File path
try:
    df = pd.read_csv(file_path)
    print("Updated dataset successfully loaded.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Step 2: Define the list of 11 countries (with spaces maintained)
countries_of_interest = [
    "USA", "CHINA", "GERMANY", "JAPAN", "UK",
    "FRANCE", "ITALY", "CANADA", "BRAZIL", "RUSSIA", "SOUTH KOREA"
]

# Normalize 'Country' column: Convert to uppercase and strip leading/trailing whitespace
df.columns = df.columns.str.upper()  # Convert all column names to uppercase
if 'COUNTRY' in df.columns:  # Ensure 'Country' column exists
    df['COUNTRY'] = df['COUNTRY'].str.upper().str.strip()  # Normalize case and remove extra spaces

    # Filter the dataset for the 11 countries
    filtered_df = df[df['COUNTRY'].isin(countries_of_interest)]
    print("\nFiltering completed successfully.")
else:
    print("Error: 'COUNTRY' column not found in the dataset.")
    exit()

# Step 3: Show all unique countries in the filtered dataset
print("\nUnique countries in the filtered dataset:")
unique_countries = filtered_df['COUNTRY'].unique()
print(unique_countries)

# Display all column names
print("\nColumns in the Dataset:")
print(filtered_df.columns)

# Step 4: Check for missing values
print("\nChecking for missing values in the filtered dataset:")
missing_values = filtered_df.isnull().sum()
print(missing_values)

# Handle missing values
filtered_df_cleaned = filtered_df.dropna()  # Drop rows with missing values
print("\nRows with missing values dropped.")
print(f"Shape of filtered dataset after dropping missing rows: {filtered_df_cleaned.shape}")

# Step 5: Remove duplicate rows
print("\nChecking for duplicate rows in the filtered dataset:")
duplicates = filtered_df_cleaned.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

filtered_df_cleaned = filtered_df_cleaned.drop_duplicates()
print("\nDuplicate rows removed.")
print(f"Shape of filtered dataset after removing duplicates: {filtered_df_cleaned.shape}")

# Step 6: Convert quantities to tons if 'QUANTITY' and 'UNIT' columns exist
if 'QUANTITY' in filtered_df_cleaned.columns and 'UNIT' in filtered_df_cleaned.columns:
    print("\nConverting all quantities to tons...")
    
    def convert_to_tons(row):
        if row['UNIT'] == 'KGS':
            return row['QUANTITY'] / 1000  # Convert kilograms to tons
        elif row['UNIT'] == 'TONS':
            return row['QUANTITY']  # Already in tons
        else:
            return row['QUANTITY']  # Keep original quantity if unknown

    # Apply conversion
    filtered_df_cleaned['QUANTITY'] = filtered_df_cleaned.apply(convert_to_tons, axis=1)
    filtered_df_cleaned['UNIT'] = 'TONS'  # Update all units to tons
    print("\nQuantities converted to tons successfully.")
else:
    print("Error: 'QUANTITY' or 'UNIT' column not found in the dataset.")
    exit()

# Step 7: Save the cleaned and standardized dataset
output_file = r"E:\Python project\filtered_cleaned_standardized_11_countries_tons.csv"
filtered_df_cleaned.to_csv(output_file, index=False)
print(f"Filtered, cleaned, and standardized dataset (in tons) saved to '{output_file}'.")

# Step 8: EDA Process
print("\nIntroduction:")
print("This dataset represents export data filtered for 11 specific countries. The dataset is cleaned, standardized to tons, and ready for analysis.")
print("Objective: Perform EDA to understand export trends, highlight key insights, and visualize results.")

# General Description
print("\nGeneral Description:")
print(f"Dataset Shape: {filtered_df_cleaned.shape}")
print("Columns in Dataset:")
print(filtered_df_cleaned.columns) 
print("\nUnique Countries in Dataset:")
print(filtered_df_cleaned['COUNTRY'].unique())
print("\n🔹 Descriptive Statistics for Numeric Columns:")
print(filtered_df_cleaned.describe().round(2)) # Summary statistics for numerical columns
print("\n🔹 Descriptive Statistics for Categorical Columns:")
print(filtered_df_cleaned.describe(include=['object']))
# Use the correct column for export value ('VALUE(US$ MILLION)')
export_value_column = 'VALUE(US$ MILLION)'  # Correct column name for export value

# Specific Requirements and Analysis
print("\nTotal Export Value by Country:")
export_value_by_country = filtered_df_cleaned.groupby('COUNTRY')[export_value_column].sum()
print(export_value_by_country)

print("\nTotal Quantity (in Tons) by Country:")
quantity_by_country = filtered_df_cleaned.groupby('COUNTRY')['QUANTITY'].sum().round(2)
print(quantity_by_country)

print("\nCorrelation Analysis:")
correlation = filtered_df_cleaned[['QUANTITY', export_value_column]].corr()
print(correlation)

# -----------------------------------------------------
# Step 9: Linear Regression - Export Quantity vs Value
# -----------------------------------------------------


# Prepare the data
X = filtered_df_cleaned[['QUANTITY']]  # Independent variable
y = filtered_df_cleaned[export_value_column]  # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\n--- Linear Regression Model Summary ---")
print(f"Intercept: {model.intercept_}")
print(f"Coefficient (Slope): {model.coef_[0]}")
print(f"R² Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Visualize the regression line on the data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['QUANTITY'], y=y, alpha=0.6, color='blue', label='Actual')
plt.plot(X['QUANTITY'], model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title("Linear Regression: Export Quantity vs Export Value")
plt.xlabel("Quantity (Tons)")
plt.ylabel("Export Value (US$ Million)")
plt.legend()
plt.tight_layout()
plt.show()


# Original Visualizations

# Visualization 1: Bar Plot - Export Value by Country
plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_df_cleaned, x='COUNTRY', y=export_value_column, errorbar=None, palette='viridis')
plt.title("Export Value by Country")
plt.xlabel("Country")
plt.ylabel("Export Value (US$ Million)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 2: Bar Plot - Quantity by Country
plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_df_cleaned, x='COUNTRY', y='QUANTITY', errorbar=None, palette='magma')
plt.title("Total Quantity (in Tons) by Country")
plt.xlabel("Country")
plt.ylabel("Quantity (Tons)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 3: Pie Chart - Export Share by Country
plt.figure(figsize=(8, 8))
export_value_by_country.plot.pie(autopct='%1.1f%%', startangle=140, cmap='Set3')
plt.title("Export Share by Country")
plt.ylabel("")  # Hide y-label for better clarity
plt.show()

# Advanced Visualizations
# Visualization 4: Top 10 Commodities by Export Value
top_commodities = filtered_df_cleaned.groupby('PRINCIPLE COMMODITY')[export_value_column].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(14, 7))
sns.barplot(x=top_commodities.values, y=top_commodities.index, palette="coolwarm")
plt.title("Top 10 Commodities by Export Value")
plt.xlabel("Export Value (US$ Million)")
plt.ylabel("Commodity")
plt.tight_layout()
plt.show()

# Visualization 5: Distribution of Export Values (KDE Plot)
plt.figure(figsize=(12, 6))
sns.kdeplot(data=filtered_df_cleaned, x=export_value_column, hue='COUNTRY', fill=True, common_norm=False, palette='coolwarm', alpha=0.7)
plt.title("Export Value Distribution (Kernel Density Plot)")
plt.xlabel("Export Value (US$ Million)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# Visualization 6: Pair Plot - Quantity and Export Value
sns.pairplot(filtered_df_cleaned, vars=['QUANTITY', export_value_column], hue='COUNTRY', palette='tab20', height=3)
plt.suptitle("Pair Plot of Quantity and Export Value by Country", y=1.02)
plt.tight_layout()
plt.show()

# Visualization 7: Heatmap of Top Countries and Commodities
top_commodities_by_country = filtered_df_cleaned.pivot_table(
    values=export_value_column,
    index='COUNTRY',
    columns='PRINCIPLE COMMODITY',
    aggfunc='sum',
    fill_value=0
)
plt.figure(figsize=(16, 10))
sns.heatmap(top_commodities_by_country, annot=False, cmap='YlGnBu', cbar_kws={'label': 'Export Value (US$ Million)'})
plt.title("Heatmap of Export Contributions by Country and Commodity")
plt.tight_layout()
plt.show()


# Visualization 8: Export Value vs. Quantity Relationship per Country (Bubble Chart)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=filtered_df_cleaned, x='QUANTITY', y=export_value_column, hue='COUNTRY', size=export_value_column, sizes=(40, 400), palette="viridis", alpha=0.7)
plt.title("Export Value vs Quantity by Country")
plt.xlabel("Quantity (Tons)")
plt.ylabel("Export Value (US$ Million)")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# Visualization 9: Country-wise Distribution of Export Value (Violin Plot)
plt.figure(figsize=(12, 7))
sns.violinplot(data=filtered_df_cleaned, x='COUNTRY', y=export_value_column, palette="Set3")
plt.title("Country-wise Distribution of Export Value")
plt.xlabel("Country")
plt.ylabel("Export Value (US$ Million)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Visualization 10: Count Plot - Frequency of Countries
plt.figure(figsize=(12, 6))
sns.countplot(data=filtered_df_cleaned, x='COUNTRY', palette="pastel")
plt.title("Frequency of Export Destinations (Count Plot)")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks


# Visualization 11: Scatter Plot - Export Value vs Quantity
plt.figure(figsize=(12, 8))
sns.scatterplot(data=filtered_df_cleaned, x='QUANTITY', y=export_value_column, hue='COUNTRY', palette='tab10')
plt.title("Export Value vs Quantity by Country (Scatter Plot)")
plt.xlabel("Quantity (Tons)")
plt.ylabel("Export Value (US$ Million)")
plt.tight_layout()
plt.show()


# Visualization 12: Histogram - Distribution of Export Values
plt.figure(figsize=(12, 6))
sns.histplot(data=filtered_df_cleaned, x=export_value_column, kde=True, color='blue', bins=20)
plt.title("Distribution of Export Values (Histogram)")
plt.xlabel("Export Value (US$ Million)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Visualization 13: Histogram of Export Quantities
plt.figure(figsize=(12, 6))
sns.histplot(data=filtered_df_cleaned, x='QUANTITY', kde=True, color='green', bins=20)
plt.title("Distribution of Export Quantities")
plt.xlabel("Export Quantity (Tons)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Visualization 14: Box Plot - Export Value Distribution per Country
plt.figure(figsize=(14, 8))
sns.boxplot(data=filtered_df_cleaned, x='COUNTRY', y=export_value_column, palette='Set2')
plt.title("Export Value Distribution per Country (Box Plot)")
plt.xlabel("Country")
plt.ylabel("Export Value (US$ Million)")
plt.xticks(rotation=45)

# Set a larger y-axis range for better visibility
plt.ylim(0, filtered_df_cleaned[export_value_column].max() + 500)  # Add padding to the max value
plt.tight_layout()
plt.show()

# Visualization 15: Heatmap - Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(filtered_df_cleaned[['QUANTITY', export_value_column]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Visualization 16: Area Plot - Export Value by Country
plt.figure(figsize=(12, 6))
export_value_by_country.sort_values().plot(kind='area', color="skyblue", alpha=0.5)
plt.title("Export Value by Country (Area Plot)")
plt.xlabel("Country")
plt.ylabel("Export Value (US$ Million)")
plt.tight_layout()
plt.show()

# Visualization 17: Strip Plot - Quantity Distribution per Country
# Strip Plot with Custom Y-Axis Limits
plt.figure(figsize=(12, 6))
sns.stripplot(data=filtered_df_cleaned, x='COUNTRY', y='QUANTITY', palette='Set3', jitter=True)
plt.title("Quantity Distribution with Adjusted Y-Axis")
plt.xlabel("Country")
plt.ylabel("Quantity (Tons)")
plt.ylim(0, 100000)  # Set the upper limit for visibility
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualization 18: Swarm Plot - Export Value per Country
plt.figure(figsize=(12, 6))
sns.swarmplot(data=filtered_df_cleaned, x='COUNTRY', y=export_value_column, palette='tab20')
plt.title("Export Value Distribution (Swarm Plot)")
plt.xlabel("Country")
plt.ylabel("Export Value (US$ Million)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("\nEDA process with advanced visualizations completed successfully.")