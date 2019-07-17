import pandas as pd

raw_data_path = '/home/mattia/Desktop/ProgettoSteve/ToyDataset/Validation/part-009.csv'

data = pd.read_csv(raw_data_path, usecols=range(5,235))

print("Original data: ", data.shape)
print(data.head())
