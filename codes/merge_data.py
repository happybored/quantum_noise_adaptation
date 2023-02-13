from re import sub
from statistics import mean
from sklearn import cluster
from sklearn.preprocessing import normalize,StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

cluster_info_path = 'calibration_data/cluster_res/ibmq_belem-tech-offline-cluster-k_k_L1-info.xlsx'
cluster_data_path = 'calibration_data/cluster_res/ibmq_belem-tech-offline-cluster-k_k_L1-result.xlsx'
# total_center_point_path = 'calibration_data/cluster_res/ibmq_belem-tech-offline-cluster-k_k_L1-info.xlsx'
np.random.seed(0)


info_df = pd.read_excel(cluster_info_path)
data_df = pd.read_excel(cluster_data_path)
# center_df = pd.read_excel(total_center_point_path)

keys = ['id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3']
keys2 = ['timestamp','id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3','accuracy','dis','label','center']
keys3 = ['label' , 'num' , 'mean_acc' , 'mean_noise' , 'max_noise' , 'min_noise' ,'center_p' , 'comp_acc']
print(info_df)
cluster_labels = list(info_df['label'])
cluster_centers = list(info_df['center_p'])
print(cluster_labels)
print(cluster_centers)
data_df['center'] = 0
for i in range(len(data_df)):
    label = data_df['label'].loc[i]
    if label in cluster_labels:
        data_df['center'].loc[i] =info_df.loc[info_df['label']==label]['center_p'].values[0]
    else:
        min_center = ''
        min_dis = 1000
        for date in cluster_centers:
            dis = np.linalg.norm(data_df[keys].loc[data_df['timestamp']==date]- data_df[keys].loc[i] ,ord=2,axis=1)[0]
            print(dis)
            if dis<min_dis:
                min_dis = dis
                min_center = date
        data_df['center'].loc[i] = min_center
print(data_df[keys2])
data_df[keys2].to_excel(cluster_data_path)

# print(center_df)

# info_df['comp_acc'] =0 
# for i in range(len(info_df)):
#     timestamp = info_df.loc[i]['center_p']
#     print(timestamp)
#     info_df['comp_acc'].loc[i] = center_df['comp_acc'].loc[center_df['timestamp']==timestamp].values[0]
# print(info_df)
# info_df[keys3].to_excel(cluster_info_path)