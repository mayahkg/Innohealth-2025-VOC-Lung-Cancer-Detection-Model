import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import shap
from Datasets.CompilingData import *


# Setting the column class as the predictor variable 
X = data.drop('Class', axis=1)  
y = data['Class']  
# Scaling the fitting data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoding the predictor variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # One-hot encoding

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)


# Training the Model
# SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

# XGBoost Model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_encoded)

# Feedforward Neural Network
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(y_categorical.shape[1], activation='softmax'))  # Output layer
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Convolutional Neural Network (CNN)
# Reshape data for CNN (assuming 1D data for simplicity)
X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

cnn_model = Sequential()
cnn_model.add(Conv2D(64, (3, 1), activation='relu', input_shape=(X_train_cnn.shape[1], 1, 1)))
cnn_model.add(MaxPooling2D(pool_size=(2, 1)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(y_categorical.shape[1], activation='softmax'))
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=1000, batch_size=32, verbose=0)

# Recurrent Neural Network (RNN)
rnn_model = Sequential()
rnn_model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
rnn_model.add(LSTM(128))
rnn_model.add(Dense(128, activation='relu'))
rnn_model.add(Dense(y_categorical.shape[1], activation='softmax'))
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape data for RNN
X_train_rnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_rnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
rnn_model.fit(X_train_rnn, y_train, epochs=1000, batch_size=32, verbose=0)

# Evaluate models
models = {
    'SVM': svm_model,
    'XGBoost': xgb_model,
    'Feedforward NN': nn_model,
    'CNN': cnn_model,
    'RNN': rnn_model
}

for name, model in models.items():
    if name == 'SVM' or name == 'XGBoost':
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_rnn if name == 'RNN' else X_test_cnn)
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print(f"{name} Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    print(f"{name} Confusion Matrix:")
    print(confusion_matrix(y_true_classes, y_pred_classes))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_classes, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    print(f"{name} ROC AUC: {roc_auc}")

# Feature Importance using Random Forest and SHAP
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_encoded)

# Generate SHAP values
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)