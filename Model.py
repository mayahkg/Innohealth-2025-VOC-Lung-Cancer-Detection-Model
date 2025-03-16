import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
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

# Create mapping dictionary for the key
unique_classes = label_encoder.classes_
encoded_values = label_encoder.transform(unique_classes)
class_mapping = dict(zip(encoded_values, unique_classes))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Training SVM Model
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, np.argmax(y_train, axis=1))

# Training the XGBoost Model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, np.argmax(y_train, axis=1))

# Training the Neural Network Model using Functional API
nn_inputs = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(nn_inputs)
x = Dense(32, activation='relu')(x)
nn_outputs = Dense(y_categorical.shape[1], activation='softmax')(x)
nn_model = Model(inputs=nn_inputs, outputs=nn_outputs)

nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

# Training the CNN using Functional API
X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1, 1)
X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1, 1)

cnn_inputs = Input(shape=(X_train_cnn.shape[1], 1, 1))
x = Conv2D(64, (3, 1), activation='relu')(cnn_inputs)
x = MaxPooling2D(pool_size=(2, 1))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
cnn_outputs = Dense(y_categorical.shape[1], activation='softmax')(x)
cnn_model = Model(inputs=cnn_inputs, outputs=cnn_outputs)

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=1000, batch_size=32, verbose=0)

# Testing
models = [("SVM", svm_model), ("XGBoost", xgb_model), ("NNM", nn_model), ("CNN", cnn_model)]
results = []

for model_name, model in models:
    if model_name == "CNN":
        y_pred = model.predict(X_test_cnn)
        y_pred = np.argmax(y_pred, axis=1)
    elif model_name in ["NNM"]:
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)

    # Convert predictions to original class labels
    pred_classes = [class_mapping[pred] for pred in y_pred]
    actual_classes = [class_mapping[act] for act in np.argmax(y_test, axis=1)]

    results.append({
        'Model': model_name,
        'Predictions': pred_classes,
        'Actual': actual_classes,
        'Accuracy': accuracy
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save as CSV with key information in the header
with open('model_results.csv', 'w') as f:
    f.write("# Key:\n")
    for encoded_val, original_class in class_mapping.items():
        f.write(f"# {encoded_val} = {original_class}\n")
    results_df.to_csv(f, index=False)

print("\nKey:")
for encoded_val, original_class in class_mapping.items():
    print(f"{encoded_val} = {original_class}")
print("\nResults:")
print(results_df)