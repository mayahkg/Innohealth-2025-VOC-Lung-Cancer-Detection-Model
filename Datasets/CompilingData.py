import pandas as pd

# Changing file format into csv
data = pd.read_csv('Datasets/LungCancer.txt', delimiter='\t')  # Adjust delimiter if needed
print(data.head())
print(data.info())

# Saving data into csv file
data.to_csv('Datasets/DataSet.csv', index=False)