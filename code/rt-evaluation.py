#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:42:57 2019

@author: mattia
"""

from matplotlib import pyplot as plt
from matplotlib import animation

import os
import re
import numpy as np
import seaborn as sns

test_data_path = '../data/matrix_data/test_data/'
reconstructed_data_path = '../data/matrix_data/reconstructed_data/'

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)

# Computing values
test_end = 2000
test_start = 1400
thred_b = 0.005

test_anomaly_score = np.zeros((test_end - test_start, 1))

anomaly_pos = np.zeros(5)
root_cause_gt = np.zeros((5, 3))
gap_time=10
anomaly_span = [10, 30, 90]
root_cause_f = open("../data/test_anomaly.csv", "r")
row_index = 0
for line in root_cause_f:
	line=line.strip()
	anomaly_axis = int(re.split(',',line)[0])
	anomaly_pos[row_index] = anomaly_axis/gap_time - test_start - anomaly_span[row_index%3]/gap_time
	root_list = re.split(',',line)[1:]
	for k in range(len(root_list)-1):
		root_cause_gt[row_index][k] = int(root_list[k])
	row_index += 1
root_cause_f.close()

ground_truth = np.zeros(test_anomaly_score.shape[0], dtype=bool)
for i in range(len(ground_truth)):
    for k in range(len(anomaly_pos)):
        if anomaly_pos[k] <= i <= (anomaly_pos[k] + anomaly_span[k%3]/10):
            ground_truth[i] = True
    
def animate(i):
    index = i + test_start
    
    path_temp_1 = os.path.join(test_data_path, "test_data_" + str(index) + '.npy')
    gt_matrix_temp = np.load(path_temp_1)
    
    path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(index) + '.npy')
    reconstructed_matrix_temp = np.load(path_temp_2)
    reconstructed_matrix_temp = np.transpose(reconstructed_matrix_temp[0], [0, 3, 1, 2])
    
    #first (short) duration scale for evaluation  
    select_gt_matrix = np.array(gt_matrix_temp)[4][0] #get last step matrix
    
    select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]
    
    #compute number of broken element in residual matrix
    select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
    num_broken = len(select_matrix_error[select_matrix_error > thred_b])
    
    test_anomaly_score[index - test_start] = num_broken
    
    # Plotting values
    sns.heatmap(
            select_gt_matrix,
            square=True,
            cmap='coolwarm',
            cbar=False,
            ax=ax0,
            vmax=0.8).set_title('Input matrix')
    sns.heatmap(
            select_reconstructed_matrix,
            square=True,
            cmap='coolwarm',
            cbar=False,
            ax=ax1,
            vmax=0.8).set_title('Reconstructed matrix')
    sns.heatmap(
            select_matrix_error,
            square=True,
            cmap='coolwarm',
            cbar=False,
            ax=ax2,
            vmax=0.01).set_title('Residual matrix')
    
    ax3.clear()
    ax3.set_xlim(0, test_end/10)
    ax3.plot(test_anomaly_score)
    time = np.arange(0, test_anomaly_score.shape[0])
    ax3.fill_between(
        time,
        0, 1,
        where=ground_truth,
        alpha=0.4,
        color='red',
        transform=ax3.get_xaxis_transform())
    ax3.title.set_text('Anomaly score %d' % num_broken)
    
anim = animation.FuncAnimation(fig, animate, frames=200, repeat=False, interval=15)
