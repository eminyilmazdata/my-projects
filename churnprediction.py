#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/salamivoormij/Documents/datasets/churn/telecom_churn_clean.csv')

# Display basic information about the dataset
print(df.info())

# Show the first few rows
print(df.head())


# In[6]:


plt.figure(figsize=(8, 5))
sns.countplot(x='churn', data=df)
plt.title('Distribution of Churn')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# In[7]:


# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# In[8]:


plt.figure(figsize=(8, 5))
sns.countplot(x='customer_service_calls', hue='churn', data=df)
plt.title('Customer Service Calls vs Churn')
plt.xlabel('Number of Customer Service Calls')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right', labels=['No', 'Yes'])
plt.show()


# In[9]:


plt.figure(figsize=(8, 5))
sns.boxplot(x='churn', y='total_day_minutes', data=df)
plt.title('Total Day Minutes by Churn')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Total Day Minutes')
plt.show()


# In[10]:


sns.pairplot(df, hue='churn', vars=['account_length', 'total_day_minutes', 'total_night_minutes', 'total_intl_minutes'])
plt.title('Pair Plot of Selected Features')
plt.show()


# In[11]:


get_ipython().system('pip install scikit-learn')


# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['international_plan', 'voice_mail_plan', 'area_code']  # Adjust as necessary

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
    
# Define features (X) and target variable (y)
X = df.drop(columns=['churn', 'Unnamed: 0'])  # Drop target and any unnecessary columns
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Random Forest classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Feature importance
importances = model.feature_importances_
features = X.columns

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()


# In[ ]:




