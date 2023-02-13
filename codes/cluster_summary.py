from re import sub
from statistics import mean
from sklearn import cluster
from sklearn.preprocessing import normalize,StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import warnings
from mlinsights.mlmodel import KMeansL1L2
import math
warnings.filterwarnings("ignore")

path  = '/home/zhirui/Software/quantum/adaptive_noise/calibration_data/cluster_res/ibmq_belem-tech-offline-cluster-k_k_L1-result.xlsx'
keys  = ['id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3']

df = pd.read_excel(path)
mean_list =[]
mean_avg =0
mean_for_sample = df['cluster_acc'].mean()
print(mean_for_sample)
for i in range(1,7):
    samples = df.loc[df['label']==i]
    acc_mean = samples['cluster_acc'].mean()
    mean_list.append(acc_mean)
    mean_avg = mean_avg + acc_mean/6


print(mean_for_sample)
print(mean_avg)
print(mean_list)

import sys
total_dis = 0
for i in range(len(df)):
    x1 = df.loc[i][keys].values
    centerT = df.loc[i]['center']
    center = df.loc[df['timestamp']==centerT][keys].values
    # print(x1)
    # print(' ---')
    # print(center)
    # sys.exit(0)
    dis = np.abs(x1-center).sum()
    total_dis = total_dis+dis
print(total_dis)
