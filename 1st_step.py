import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('Cleaned_healthcare.csv')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {df.shape}")
print(f"Number of patients: {len(df)}")
print(f"Number of features: {len(df.columns)}")

print("\n=== BASIC INFORMATION ===")
print(df.info())

print("\n=== FIRST FEW ROWS ===")
print(df.head())

print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe())

# Medical Condition Analysis
plt.figure(figsize=(15, 10))

# Plot 1: Medical Condition Distribution
plt.subplot(2, 3, 1)
condition_counts = df['Medical Condition'].value_counts()
plt.bar(condition_counts.index, condition_counts.values)
plt.title('Distribution of Medical Conditions')
plt.xticks(rotation=45)
plt.ylabel('Number of Patients')

# Plot 2: Gender Distribution
plt.subplot(2, 3, 2)
gender_counts = df['Gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values)
plt.title('Gender Distribution')
plt.ylabel('Number of Patients')

# Plot 3: Smoking Status
plt.subplot(2, 3, 3)
smoking_counts = df['Smoking'].value_counts()
plt.bar(['Non-Smoker', 'Smoker'], smoking_counts.values)
plt.title('Smoking Status Distribution')
plt.ylabel('Number of Patients')

# Plot 4: Alcohol Consumption
plt.subplot(2, 3, 4)
alcohol_counts = df['Alcohol'].value_counts()
plt.bar(['Non-Drinker', 'Drinker'], alcohol_counts.values)
plt.title('Alcohol Consumption')
plt.ylabel('Number of Patients')

# Plot 5: Medical Condition by Gender
plt.subplot(2, 3, 5)
condition_gender = pd.crosstab(df['Medical Condition'], df['Gender'])
condition_gender.plot(kind='bar', ax=plt.gca())
plt.title('Medical Conditions by Gender')
plt.xticks(rotation=45)
plt.ylabel('Number of Patients')

plt.tight_layout()
plt.show()

# Detailed statistics
print("\n=== CATEGORICAL VARIABLE SUMMARY ===")
print("Medical Conditions:")
print(condition_counts)
print(f"\nMost common condition: {condition_counts.index[0]} ({condition_counts.iloc[0]} patients)")
print(f"Least common condition: {condition_counts.index[-1]} ({condition_counts.iloc[-1]} patients)")

print(f"\nGender Distribution:")
print(gender_counts)
print(f"Gender ratio (M/F): {gender_counts['Male']/gender_counts['Female']:.2f}")

print(f"\nSmoking Status:")
print(f"Smokers: {smoking_counts[1]} ({smoking_counts[1]/len(df)*100:.1f}%)")
print(f"Non-smokers: {smoking_counts[0]} ({smoking_counts[0]/len(df)*100:.1f}%)")

# Boxplots for key metrics across medical conditions
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# BMI by Condition
df.boxplot(column='BMI', by='Medical Condition', ax=axes[0,0])
axes[0,0].set_title('BMI Distribution by Medical Condition')
axes[0,0].tick_params(axis='x', rotation=45)

# Glucose by Condition
df.boxplot(column='Glucose', by='Medical Condition', ax=axes[0,1])
axes[0,1].set_title('Glucose Levels by Medical Condition')
axes[0,1].tick_params(axis='x', rotation=45)

# Blood Pressure by Condition
df.boxplot(column='Blood Pressure', by='Medical Condition', ax=axes[1,0])
axes[1,0].set_title('Blood Pressure by Medical Condition')
axes[1,0].tick_params(axis='x', rotation=45)

# Age by Condition
df.boxplot(column='Age', by='Medical Condition', ax=axes[1,1])
axes[1,1].set_title('Age Distribution by Medical Condition')
axes[1,1].tick_params(axis='x', rotation=45)

plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.show()

# OLS Regression Model
# Define the predictors (X) and target (y)
X = df[['Age', 'Glucose', 'Blood Pressure', 'BMI', 'Cholesterol', 'HbA1c']]  # Predictor variables
y = df['LengthOfStay']  # Target variable

# Add a constant to the predictors matrix (intercept term)
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X).fit()

# View the summary of the regression results
model_summary = model.summary()
print(model_summary)