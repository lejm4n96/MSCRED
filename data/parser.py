import pandas as pd
import numpy as np

raw_data_path = 'part-009.csv'

raw_data = pd.read_csv(raw_data_path, usecols=range(4, 235))

print("Original data: ", raw_data.shape)
# print(raw_data.head())

# delete columns with null standard deviation
null_std = (raw_data.std(axis=0) == 0)
data = raw_data.loc[:, raw_data.std() != 0.0]

# min-max normalization
data = np.array(data, dtype=np.float64)
sensor_n = data.shape[1]

max_value = np.max(data, axis=0)
min_value = np.min(data, axis=0)

data = (data - min_value)/(max_value - min_value + 1e-6)

print(data.shape)
