import pandas as pd

raw_data_path = 'synthetic_data_with_anomaly-s-1.csv'
destination_path = 'transposed_synthetic_data_with_anomaly-s-1.csv'

data = pd.read_csv(raw_data_path, header = None)

print("Original data: ", data.shape)

transposed_data = data.transpose()

print("Transposed data: ", transposed_data.shape)

transposed_data.to_csv(destination_path, header=False, index=False, float_format='%g')
