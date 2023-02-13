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

database_data_path = 'calibration_data/cluster_res_iris/ibmq_belem-tech-offline-cluster-database.xlsx'
online_data_path = 'calibration_data/cluster_res_iris/ibmq_belem-tech-online-iris-dis.xlsx'
online_res_path = 'calibration_data/cluster_res_iris/ibmq_belem-tech-online-iris-res.xlsx'
np.random.seed(0)


database_df = pd.read_excel(database_data_path)
online_df = pd.read_excel(online_data_path)

keys = ['id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3']
keys1 =['timestamp','id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3']
keys2 = ['timestamp','id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3','center','noise']
database_df = database_df[keys1]
print(database_df)
cluster_centers = list(database_df['timestamp'])
cluster_centers_noise = dict()
bad_cluseter = list()
bad_cluseter.append('2022-03-20 23:20:07-04:00')
print(cluster_centers)
online_df['center'] = 0
online_df['comp_acc'] = 0
online_df['train_flag'] = 0
online_df['error_flag'] = 0
online_df['noise'] = 0

for date in cluster_centers:
    noise = np.linalg.norm(database_df[keys].loc[database_df['timestamp']==date] ,ord=1)
    cluster_centers_noise[date] = noise
print(cluster_centers_noise)
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

    online_df['center'].loc[i] = min_center
print(online_df[keys2])
print(cluster_centers)
print(bad_cluseter)
online_df[keys2].to_excel(online_res_path)