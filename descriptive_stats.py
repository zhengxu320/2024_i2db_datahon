import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Read CSV file
data = pd.read_csv('Training_Set.csv')

# Get all column names except the first one
columns = data.columns[1:]

# Define the threshold for outliers (1.5 times the interquartile range)
def is_outlier(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ~col.between(lower_bound, upper_bound)

# Create a dictionary to store the missing value counts for each variable
missing_counts = {}

# Create the 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Perform descriptive statistics for each column
output = ""
for col in columns:
    missing_count = data[col].isnull().sum()
    missing_counts[col] = missing_count

    valid_count = len(data[col]) - missing_count
    valid_data = data[col].dropna()

    output += f"Variable: {col}\n"
    output += f"Missing Values Count: {missing_count}, Valid Data Count: {valid_count}\n"

    if valid_data.dtype == 'float64' or valid_data.dtype == 'int64':
        output += "Data Type: Continuous Variable\n"
        output += f"Mean: {valid_data.mean():.2f}\n"
        output += f"Median: {valid_data.median():.2f}\n"
        output += f"25% Quantile: {valid_data.quantile(0.25):.2f}\n"
        output += f"75% Quantile: {valid_data.quantile(0.75):.2f}\n"

        outlier_count = is_outlier(valid_data).sum()
        output += f"Outlier Count: {outlier_count}\n"

        # Plot histogram for continuous variable
        plt.figure()
        plt.hist(valid_data, bins='auto')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'plots/{col}_hist.png')
        plt.close()

    else:
        output += "Data Type: Categorical Variable\n"
        output += "Categories and Proportions:\n"

        category_counts = valid_data.value_counts()
        category_percentages = category_counts / valid_count * 100

        for category, count, percentage in zip(category_counts.index, category_counts.values, category_percentages.values):
            output += f"{category}: Count {count}, Percentage {percentage:.2f}%\n"

        # Plot pie chart for categorical variable
        plt.figure()
        plt.pie(category_counts, labels=category_counts.index, autopct='%.1f%%')
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'plots/{col}_pie.png')
        plt.close()

    output += "------------------------\n"

# Sort variables based on missing value counts
sorted_columns = sorted(missing_counts, key=missing_counts.get, reverse=True)

# Write the sorted descriptive statistics to a txt file
with open('descriptive_stats.txt', 'w') as file:
    file.write(f"Total Variables: {len(columns)}\n")
    file.write(f"Total Observations: {len(data)}\n\n")
    file.write("Variables Sorted by Missing Value Counts:\n\n")
    for col in sorted_columns:
        file.write(f"Variable: {col}\n")
        file.write(f"Missing Value Count: {missing_counts[col]}\n")
        file.write("------------------------\n")
    file.write("\nDetailed Descriptive Statistics:\n\n")
    file.write(output)