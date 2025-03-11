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
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)  # Fit and transform

# Print a mapping of the encoded values to the original class names
print("Encoded classes mapping:")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"{index}: {class_name}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

# Training SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

# Testing SVM Model
y_pred = svm_model.predict(X_test)

# Convert the predicted labels back to original class names
y_pred_original = label_encoder.inverse_transform(y_pred)
y_test_original = label_encoder.inverse_transform(y_test)

# Create a DataFrame to display the actual vs predicted values
results = pd.DataFrame({
    'Actual': y_test_original,
    'Predicted': y_pred_original
})

# Print the results in a table-like structure
print(results)

# Print the accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))