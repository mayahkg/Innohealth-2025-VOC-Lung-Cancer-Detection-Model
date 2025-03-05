import tensorflow
from sklearn.model_selection import train_test_split
from Datasets.CompilingData import *

x = data.drop('target_column', axis=1)
y = data['target_column']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)