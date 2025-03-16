import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from Datasets.CompilingData import *

# Load the dataset
data = pd.read_csv('Datasets/LungCancer.txt', delimiter='\t') 
X = data.drop('Class', axis=1)  
Y = data['Class']  

# Setting up the X and Y values
X_scaled = StandardScaler().fit_transform(X)
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)  

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

# Training SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

# Training the XGBoost Model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)  

# Testing
models = [("SVM", svm_model), ("XGBoost", xgb_model)]
results = []

for model_name, model in models:
    if model_name == "SVM" or "XGBoost":
        y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    accuracy = accuracy_score(y_pred, y_test)

    # Store results
    results.append({
        'Model': model_name,
        'Predictions': list(y_pred),
        'Actual': list(y_test),
        'Accuracy': accuracy
    })

# Framing Results
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('model_results.csv', index=False)