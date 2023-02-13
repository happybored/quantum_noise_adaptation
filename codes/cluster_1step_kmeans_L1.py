#input: offline data
#output: 
# 1. cluster and mean point
# 2. offline data ,label and mean_point 
# if the data is outlier, it will select the nearest to run in offline 
# if the data is labeled, it will be the label in offline

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

np.random.seed(0)
keys = ['id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3',]
path = 'calibration_data/ibmq_belem-tech-offline.xlsx'

cluster_name = 'kmeans_L1'
cluster_info_path = 'calibration_data/cluster_res/ibmq_belem-tech-offline-cluster-{}-info.xlsx'.format(cluster_name)
cluster_res_path = 'calibration_data/cluster_res/ibmq_belem-tech-offline-cluster-{}-result.xlsx'.format(cluster_name)


df = pd.read_excel(path)
df2 = df[keys]
df2['timestamp'] = df['timestamp']
df2['accuracy'] = df['accuracy-diff-good']

correlation_dict = dict()
#scale the feature, and calculate the distance 
weights = []
for key in keys:
    corr = df2[key].corr(df2['accuracy'])
    correlation_dict[key] = corr
    weight = pow (1-corr,4)
    weights.append(weight)
    df2[key] = df2[key]* weight
#verify our metrix
df2['dis'] = np.linalg.norm(df2[keys],ord=1,axis=1)
corr = df2['dis'].corr(df2['accuracy'])
print(corr)

# define the first cluster
X = df2[keys]
clus = KMeansL1L2(6, norm='L1')
clus.fit(X)
label = np.reshape(clus.labels_, len(df2))
df2['label']= label

clusters= []
for i in range(np.unique(label).size):
    cluster_dict = dict()
    samples = df2.loc[label==i]
    acc_avg =  samples['accuracy'].mean()
    cluster_dict['label'] = i
    cluster_dict['num'] = len(samples)
    cluster_dict['mean_acc'] = acc_avg
    cluster_dict['mean_noise'] = samples['dis'].mean()
    cluster_dict['max_noise'] = samples['dis'].max()
    cluster_dict['min_noise'] = samples['dis'].min()
    mean_sample = samples[keys].mean(axis=0)
    samples['in_class_dis'] = np.linalg.norm(samples[keys]-mean_sample,ord=1,axis=1)
    samples = samples.sort_values(by='in_class_dis')
    cluster_dict['center_p'] = samples['timestamp'].iloc[0]
    clusters.append(cluster_dict)
for clus_info in clusters:
    print(clus_info)

df_cluster = pd.DataFrame(clusters)
df_cluster = df_cluster.sort_values(by='num',ascending = False)
df_cluster = df_cluster.drop(df_cluster[df_cluster['num']<6].index)

print(df_cluster)
df_cluster.to_excel(cluster_info_path)
df2.to_excel(cluster_res_path)