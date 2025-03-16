import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('Datasets/LungCancer.txt', delimiter='\t') 
X = data.drop('Class', axis=1)  
Y = data['Class']    

# Setting up the X and Y values
X_scaled = StandardScaler().fit_transform(X)
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)  
y_categorical = to_categorical(Y_encoded)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Training SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, np.argmax(y_train, axis=1))  # Use class indices for SVM

# Training the XGBoost Model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, np.argmax(y_train, axis=1))  # Use class indices for XGBoost

# Training the NNM
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(y_categorical.shape[1], activation='softmax'))  # Output layer
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# Testing
models = [("SVM", svm_model), ("XGBoost", xgb_model), ("nn_model", xgb_model)]
results = []

for model_name, model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)

    results.append({
        'Model': model_name,
        'Predictions': list(y_pred),
        'Actual': list(np.argmax(y_test, axis=1)),
        'Accuracy': accuracy
    })

# Framing Results
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('model_results.csv', index=False)