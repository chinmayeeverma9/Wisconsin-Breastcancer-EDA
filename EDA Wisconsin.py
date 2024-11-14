# Importing the necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading data with a correct file path
data_path = r'C:\Users\Admin\Desktop\Chinmayee Verma\data.csv'
data = pd.read_csv(data_path)

# Display the first few rows to ensure data is loaded correctly
print(data.head())

# Step 1: Data Cleaning
# Drop unnecessary columns ('id' and 'Unnamed: 32')

data_cleaned = data.drop(columns=['id', 'Unnamed: 32'])

# Check for missing values
missing_values = data_cleaned.isnull().sum()
print("\nMissing values in each column:")
print(missing_values[missing_values > 0])

# Step 2: Analyze Class Distribution
class_distribution = data_cleaned['diagnosis'].value_counts()
print("\nClass distribution of the target variable:")
print(class_distribution)

# Step 3: Generate Descriptive Statistics
print("\nDescriptive statistics of the dataset:")
print(data_cleaned.describe())

# Step 4: Visualize Data

# Histograms of features with adjusted spacing
data_cleaned.drop(columns=['diagnosis']).hist(bins=20, figsize=(15, 15), color='skyblue')

# Adjust subplot parameters to create more space

plt.subplots_adjust(hspace=1 , wspace=1)  # hspace: height space, wspace: width space

plt.suptitle('Feature Distributions')
plt.show()

# Box plots for each feature grouped by diagnosis
plt.figure(figsize=(20, 15))
for i, column in enumerate(data_cleaned.columns[1:], 1):
    plt.subplot(6, 5, i)
    sns.boxplot(x='diagnosis', y=column, data=data_cleaned, palette="Set2")
    plt.title(f'Boxplot of {column} by Diagnosis')

# Adjust the spacing between plots
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # hspace: height space, wspace: width space

plt.show()