# House Price Prediction using Machine Learning

## 📌 Overview
This project builds a **Machine Learning model** to predict house prices based on various features like median income, house age, and location. The dataset used is the **California Housing Dataset**, available in Scikit-learn.

The project follows a **structured pipeline**:
- **Data Preprocessing**: Handling missing values, feature scaling
- **Model Training**: Using **Random Forest Regressor**
- **Model Evaluation**: Using MAE, MSE, RMSE, and R² Score
- **Model Versioning**: Automating model storage with version numbers

## 📂 Dataset
- **Dataset Source**: [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
- **Features:**
  - `MedInc` (Median Income)
  - `HouseAge` (House Age in Years)
  - `AveRooms` (Average Rooms per Household)
  - `AveBedrms` (Average Bedrooms per Household)
  - `Population` (Population of the Area)
  - `AveOccup` (Average Occupants per Household)
  - `Latitude` (Geographical Latitude)
  - `Longitude` (Geographical Longitude)
- **Target Variable:** `Price` (Median house value in USD)

## 🚀 Technologies Used
- **Python** (Jupyter Notebook)
- **Scikit-Learn** (Machine Learning and Data Preprocessing)
- **Pandas & NumPy** (Data Handling)
- **Matplotlib & Seaborn** (Data Visualization)
- **Pickle** (Model Saving)
