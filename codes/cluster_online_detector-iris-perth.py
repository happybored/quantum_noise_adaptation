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
import warnings
warnings.filterwarnings("ignore")

# database_data_path = 'calibration_data/ibmq_belem-tech-offline-database.xlsx'
# online_data_path = 'calibration_data/ibmq_belem-tech-online.xlsx'
# online_res_path = 'calibration_data/ibmq_belem-tech-offline-online.xlsx'
# database_data_path = 'calibration_data/cluster_res_iris_perth/ibmq_perth-tech-offline-cluster-database.xlsx'
# online_data_path = 'calibration_data/cluster_res_iris_perth/ibmq_perth-tech-online.xlsx'
# online_res_path = 'calibration_data/cluster_res_iris_perth/ibmq_perth-tech-only-online.xlsx'


database_data_path = 'calibration_data/cluster_res_iris_perth/ibmq_perth-tech-only-online-cluster-database.xlsx'
online_data_path = 'calibration_data/real_device/ibm_perth-2022-11-16.xlsx'
online_res_path = 'calibration_data/real_device/ibm_perth-2022-11-16-only-online.xlsx'
np.random.seed(0)


database_df = pd.read_excel(database_data_path)
online_df = pd.read_excel(online_data_path)

keys = ['id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3']
keys1 =['timestamp','id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3']
keys2 = ['timestamp','id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3','center','train_flag','error_flag','noise']
database_df = database_df[keys1]
print(database_df)


cluster_centers = list()
bad_cluseter = list()
cluster_centers_noise = dict()

# offline
cluster_centers = list(database_df['timestamp'])
print(cluster_centers)
# bad_cluseter.append()
for date in cluster_centers:
    noise = np.linalg.norm(database_df[keys].loc[database_df['timestamp']==date] ,ord=1)
    cluster_centers_noise[date] = noise




print(cluster_centers_noise)


online_df['center'] = 0
online_df['comp_acc'] = 0
online_df['train_flag'] = 0
online_df['error_flag'] = 0
online_df['noise'] = 0


for i in range(len(online_df)):
    min_center = ''
    comp_acc =0
    min_dis = 1000
    for date in cluster_centers:
        dis = np.linalg.norm(database_df[keys].loc[database_df['timestamp']==date]- online_df[keys].loc[i] ,ord=1,axis=1)[0]
        if dis<min_dis:
            min_dis = dis
            min_center = date
    noise = np.linalg.norm( online_df[keys].loc[i] ,ord=1)
    online_df['noise'].loc[i] = noise

    if min_center in bad_cluseter:
        online_df['error_flag'].loc[i] = 1
    
    if min_center == '' or min_dis > cluster_centers_noise[min_center] :
        print(len(cluster_centers))
        if noise > 0.1:
            online_df['error_flag'].loc[i] = 1
            bad_cluseter.append(online_df['timestamp'].loc[i])
        database_df.loc[len(database_df.index)] = online_df[keys1].loc[i]
        online_df['train_flag'].loc[i] = 1
        cluster_centers.append(online_df['timestamp'].loc[i])
        cluster_centers_noise[online_df['timestamp'].loc[i]] = noise
        online_df['center'].loc[i] = online_df['timestamp'].loc[i]
    # comp_acc = database_df['cluster_acc'].loc[database_df['timestamp']==min_center].values
    else:
        online_df['center'].loc[i] = min_center
    # online_df['comp_acc'].loc[i] = comp_acc
print(online_df[keys2])
print()
print(bad_cluseter)
print(cluster_centers)
print(len(cluster_centers))
online_df[keys2].to_excel(online_res_path)