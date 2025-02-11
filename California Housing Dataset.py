#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd  # For data handling
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
import seaborn as sns  # For better plots
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Data preprocessing
from sklearn.ensemble import RandomForestRegressor  # Machine Learning Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Model evaluation
import pickle  # For saving the model
import os  # For handling file paths
from sklearn.datasets import fetch_california_housing  # Load dataset


# In[2]:


# Load the dataset from Scikit-learn
data = fetch_california_housing()
df = pd.DataFrame(data=data.data, columns=data.feature_names)  # Convert to Pandas DataFrame


# In[3]:


# Add target variable (House prices)
df["Price"] = data.target  


# In[4]:


# Display the first few rows of the dataset
print(df.head())


# In[5]:


# Check for missing values
print(df.isnull().sum())  # There are no missing values in this dataset, but this is a good practice.


# In[6]:


# Split data into features (X) and target variable (y)
X = df.drop(columns=["Price"])  # Features
y = df["Price"]  # Target variable (House price)


# In[7]:


# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Standardize numerical features to improve model performance
scaler = StandardScaler()  # Create a scaler object
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)  # Only transform test data


# In[9]:


# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Train the model


# In[10]:


# Predict on test data
y_pred = model.predict(X_test)


# In[11]:


# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared Score


# In[12]:


# Print evaluation metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-Squared Score: {r2:.2f}")


# In[13]:


# Implement model versioning by checking existing models and assigning a version number
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist


# In[14]:


# Find the latest version of the model
existing_versions = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(model_dir) if f.startswith("house_price_model_v")]
new_version = max(existing_versions) + 1 if existing_versions else 1  # Assign next version number


# In[15]:


# Save the trained model as a pickle (.pkl) file
model_filename = os.path.join(model_dir, f"house_price_model_v{new_version}.pkl")
with open(model_filename, "wb") as f:
    pickle.dump(model, f)


# In[16]:


print(f"Model saved as {model_filename}")

