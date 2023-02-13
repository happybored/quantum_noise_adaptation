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
from mlinsights.mlmodel import KMeansL1L2
np.random.seed(0)

root_cluster_res_path = 'calibration_data/cluster_res/ibmq_belem-offline-'
clusters_num = [3]
for j in range(1):
    keys = ['id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3']
    df = pd.read_excel(root_cluster_res_path + str(j) + '.xlsx')
    X = df[keys]
    clus = KMeansL1L2(clusters_num[j], norm='L1')
    
    clus.fit(X)
    label = np.reshape(clus.labels_, len(df))
    print(f"Number of points: {label.size}")
    print(f"Number of clusters: {np.unique(label).size}")
    print(label)
    
    clusters= []
    for i in range(np.unique(label).size):
        cluster_dict = dict()
        samples = df.loc[label==i]
        acc_avg =  samples['accuracy'].mean()
        cluster_dict['label'] = i
        cluster_dict['num'] = len(samples)
        cluster_dict['mean_acc'] = acc_avg
        cluster_dict['mean_noise'] = samples['dis'].mean()
        cluster_dict['max_noise'] = samples['dis'].max()
        mean_sample = samples[keys].mean(axis=0)
        samples['in_class_dis'] = np.linalg.norm(samples[keys]-mean_sample,ord=1,axis=1)
        samples = samples.sort_values(by='in_class_dis')
        cluster_dict['mean'] = samples['timestamp'].iloc[0]
        clusters.append(cluster_dict)
        
    for clus_info in clusters:
        print(clus_info)
    
    df_cluster = pd.DataFrame(clusters)
    df_cluster = df_cluster.sort_values(by='num',ascending = False)
    df_cluster.to_excel(root_cluster_res_path +'cluster' +str(j) + '.xlsx')
