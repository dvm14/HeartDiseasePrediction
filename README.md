# Heart Disease Prediction

This project aims to predict the likelihood of heart disease based on clinical and lifestyle attributes using the **Heart Disease Dataset (`heart.csv`)**. The model was developed using Python with a focus on data cleaning, feature engineering, and classification using **Logistic Regression**.

---

## Dataset Overview
We used the **`heart.csv`** dataset containing several health-related features such as:
- Age, Sex, Resting Blood Pressure (RestingBP)
- Serum Cholesterol (Cholesterol)
- Fasting Blood Sugar, Max Heart Rate
- Chest Pain Type, ECG Results, Exercise-Induced Angina
- Target (indicating presence of heart disease)

---

## Data Preprocessing
1. **Data Inspection** – Checked dataset shape, column types, and summary statistics.  
2. **Missing Values** – Verified that there were *no null values* in the dataset.  
3. **Zero-Value Analysis** – Investigated whether `0` values in features were valid.  
   - Found that `RestingBP` and `Cholesterol` cannot realistically be `0`.  
   - Replaced zeros in these columns with `NaN` to mark them as missing.  
4. **Feature Encoding** – Encoded categorical features appropriately for model input.  
5. **Feature Scaling** – Applied `StandardScaler` to normalize numerical features.  
6. **Feature Selection** – Used all columns as input features except the target variable.

---

## Model Training
- **Algorithm:** Logistic Regression  
- **Train-Test Split:** Split data into training and testing sets.  
- **Evaluation Metric:** Accuracy Score  

After training, the **model achieved an accuracy of 0.864 (86.4%)** on the test set.

---

## Key Learnings
- Data cleaning and feature inspection significantly impact model performance.  
- Checking for invalid zero values is essential, especially in medical datasets.  
- Logistic Regression provides strong baseline performance for binary classification.

---

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn)
- Jupyter Notebook / Google Colab
- Matplotlib & Seaborn (optional for visualization)

---

## 📈 Future Improvements
- Experiment with advanced models like Random Forest, XGBoost, or Neural Networks.  
- Perform hyperparameter tuning for improved accuracy.  
- Handle missing values using imputation instead of direct removal.

---

## Authors
**Tiffany Degbotse**  
**Diya Mirji**
*Heart Disease Prediction using Logistic Regression — 2025*
