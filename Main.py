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

# Scale the features
X_scaled = StandardScaler().fit_transform(X)

# Encode the target variable
Y_encoded = LabelEncoder().fit_transform(Y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

# Training SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)  # Use Y_encoded for training

# Testing SVM Model
y_pred = svm_model.predict(X_test)

# Create a DataFrame to display the actual vs predicted values
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Print the results in a table-like structure
print(results)

# Print the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))