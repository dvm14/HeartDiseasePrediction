# Heart Disease Prediction

This project aims to predict the likelihood of heart disease based on clinical and lifestyle attributes using the **Heart Disease Dataset (`heart.csv`)**. The model was developed using Python with a focus on data cleaning, feature engineering, and classification using Logistic Regression and Random Forest Classifier.

The goals of this project is to practice creating an end-to-end ML pipeline:
- Practice sourcing, cleaning, and hosting real-world data in the cloud
- Build reproducible, modular ML pipelines using configuration files and containers
- Track experiments using MLFlow or Weights & Biases
- Deploy a model as an API endpoint using cloud services
- Build a lightweight front-end to consume a live model API
- Follow GitHub best practices for collaboration and version control
- Reflect on ethical implications and limitations of your ML application

---

## Dataset Overview
We used the **`heart.csv`** dataset containing several health-related features such as:
- Age, Sex, Resting Blood Pressure (RestingBP)
- Serum Cholesterol (Cholesterol)
- Fasting Blood Sugar, Max Heart Rate
- Chest Pain Type, ECG Results, Exercise-Induced Angina
- Target (indicating presence of heart disease)

The dataset in held on Google Cloud Storage: https://storage.googleapis.com/heartdiseaseprediction_bucket/data_folder/preprocessed_data.csv.

---
```
## Repository Structure

HeartDiseasePrediction/
├── data/
│   ├── heart.csv             # raw data
│   └── preprocessed_data.csv # pre-processed data
├── models/
│   ├── best_rf_model.pkl     # Best Random Forest Model (algorithm 3)
│   ├── lr_model.pkl          # Basic Logistic Regression Model (algorithm 1)
│   └── rf_model.pkl          # Basic Random Forest Model (algorithm 2)
├── src/
│   ├── app.py                # API
│   ├── config.yaml           # yaml for pipeline attributes
│   ├── main.py               # Hugging Face front-end
│   ├── process_data.py       # Pre-process the raw data
│   ├── run_model.py          # Runs training of the different models and their evaluation
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```
---
## Model Training
- **Algorithm:** Logistic Regression  
- **Train-Test Split:** Split data into training 80% and testing 20% sets.  
- **Evaluation Metric:** Accuracy Score
After training, the **model achieved an accuracy of 0.864 (86.4%)** on the test set.

-**Algorithm 2:** Random Forest (Set parameters from config.yaml)
- **Train-Test Split:** Split data into training 80% and testing 20% sets.  
- **Evaluation Metric:** Accuracy Score
  
After training, the **model achieved an accuracy of 0.826 (82.6%)** on the test set.

-**Algorithm 3:** Random Forest (Grid Search to get best parameters)
- **Train-Test Split:** Split data into training 80% and testing 20% sets.  
- **Evaluation Metric:** Accuracy Score
  
After training, the **model achieved an accuracy of 0.875 (87.5%)** on the test set.
We decided to use Algorithm 3 of Random Forest with the best parameters as our selected trained model for the deployed API.

---

## Setup

### 1) Clone the repo
```bash
git clone https://github.com/dvm14/HeartDiseasePrediction.git
cd HeartDiseasePrediction
```

### 2) Build the docker image
Build only once
```bash
docker build -t heart-disease .
```

### 3) Run training with Weights and Biases logging
```bash
docker run -e WANDB_API_KEY=<your_key> heart-disease python src/run_model.py
```

### 4) Run API server
```bash
docker run -p 8080:8080 heart-disease
http://localhost:8080 should work after running the api server
```

### Optional: Pre-process data
The data is already pre-processed and stored in a Google Cloud Storage bucket. But in case you would like to pre-process again.
```bash
docker run heart-disease python src/process_data.py
```

---

## Usage

On the front-end app, you will be able to enter patient information.
Once you click the "Predict" button at the bottom, you will receive a prediction from our trained model of whether the patient with have Heart Disease or No Heart Disease.


---

## Link to deployed API and front-end app

Deployed API: https://heart-api-41967139984.us-central1.run.app/docs

Front-end App: https://huggingface.co/spaces/tiffany101/heart_disease_predictor

---

## 📈 Future Improvements
- Experiment with advanced models like XGBoost or Neural Networks.  
- Perform hyperparameter tuning for improved accuracy.  
- Handle missing values using imputation instead of direct removal.

---

## Authors
**Tiffany Degbotse**  
**Diya Mirji**
*Heart Disease Prediction using Logistic Regression — 2025*
