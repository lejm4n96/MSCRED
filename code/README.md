# MSCRED

A demo code (tensorflow version) of MSCRED in AAAI2019 paper: A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data 
Contact: Chuxu Zhang (czhang11@nd.edu)
Notice: Although it is only a demo code, it should also be confidential. 
This is a requirement of NEC Labs America. 

### How to use
1. Generate signature matrices (train/test matrices) by running `python3 matrix_generator.py`
2. Model training by running `python3 MSCRED_TF.py --train_test_label 1`
3. Model testing by running `python3 MSCRED_TF.py --train_test_label 0`
4. After running model test, model will return recontructed matrices of test period, then evaluate model performance by running `python3 evaluate.py`

### Data
synthetic_data_with_anomaly-s-1.csv: sample time series raw data
test_anomaly.csv: anomaly position and root causes of sample data

Please remember to cite our paper. 
