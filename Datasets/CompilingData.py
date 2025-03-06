import pandas as pd

# Changing file format into csv
data = pd.read_csv('Datasets/LungCancer.txt', delimiter='\t')  # Adjust delimiter if needed
print(data.head())
print(data.info())

# Saving data into csv file
data.to_csv('Datasets/DataSet.csv', index=False)

# Data Analysis:
Control_Class = data[data['Class'] == 'Control']
Benign_Class = data[data['Class'] == 'Benign']
Cancer_Class = data[data['Class'] == 'Cancer']
print(Control_Class.describe())
print(Benign_Class.describe())
print(Cancer_Class.describe())

# Splitting the Data based of their Class
Control_Class.to_csv('Datasets/ControlSet.csv', index=False)
Benign_Class.to_csv('Datasets/BenignSet.csv', index=False)
Cancer_Class.to_csv('Datasets/CancerSet.csv', index=False)