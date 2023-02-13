import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

path = 'calibration_data/cluster_res/ibmq_belem-tech-offline.xlsx'
output_path = 'analysis_pic/'

df = pd.read_excel(path)
keys = ['id0', 'id1', 'id2', 'id3', 'id4', 'cx4_3', 'cx3_4', 'cx3_1', 'cx1_3', 'cx2_1', 'cx1_2', 'cx1_0',
       'cx0_1', 'T1_q0', 'T2_q0', 'frequency_q0', 'readout_error_q0',
       'prob_meas0_prep1_q0', 'prob_meas1_prep0_q0', 'T1_q1', 'T2_q1',
       'frequency_q1', 'readout_error_q1', 'prob_meas0_prep1_q1',
       'prob_meas1_prep0_q1', 'T1_q2', 'T2_q2', 'frequency_q2',
       'readout_error_q2', 'prob_meas0_prep1_q2', 'prob_meas1_prep0_q2',
       'T1_q3', 'T2_q3', 'frequency_q3', 'readout_error_q3',
       'prob_meas0_prep1_q3', 'prob_meas1_prep0_q3', 'T1_q4', 'T2_q4',
       'frequency_q4', 'readout_error_q4', 'prob_meas0_prep1_q4',
       'prob_meas1_prep0_q4']


corre_dict = dict()

for key in keys:
    r = df['accuracy-diff-good'].corr(df[key])
    corre_dict[key]=r
# plt.scatter(df[key],df['accuracy'])
    # plt.savefig(output_path+key +'.jpg' )
    # plt.cla()
# plt.show()

corre_dict = {k: v for k, v in sorted(corre_dict.items(), key=lambda item: item[1],reverse= True)}    

for key,val in corre_dict.items():
    print('{}\'s corr = {}'.format(key,val))

