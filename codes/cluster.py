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
path = 'calibration_data/ibmq_belem-tech-offline.xlsx'
root_cluster_res_path = 'calibration_data/cluster_res/ibmq_belem-offline-'
distan_res_path = 'calibration_data/ibmq_belem-tech-offline-distance-result.xlsx'
cluster_res_path = 'calibration_data/ibmq_belem-tech-offline-cluster-result.xlsx'

np.random.seed(0)


df = pd.read_excel(path)
keys = [ 'cx4_3', 'cx3_4', 'cx3_1', 'cx1_3', 'cx2_1', 'cx1_2', 'cx1_0',
       'cx0_1','accuracy']
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

keys = ['id0', 'id1', 'id2', 'id3', 'cx0_1','cx1_2','cx1_3',]

def calculateMahalanobis(y=None, data=None, cov=None):
    y_mu = y-np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()


df2 = df[keys]
df2['accuracy'] = df['accuracy-diff-good']

correlation_dict = dict()
weights = []
for key in keys:
    corr = df2[key].corr(df2['accuracy'])
    correlation_dict[key] = corr
    weight = pow (1-corr,4)
    weights.append(weight)
    df2[key] = df2[key]* weight

# df2['Mahalanobis'] = calculateMahalanobis(y=df2[keys], data=df2[keys])
df2['dis'] = np.linalg.norm(df2[keys],ord=1,axis=1)
df2['p'] = 1 - stats.chi2.cdf(df2['dis'], 5)
print(df2)
print(correlation_dict)

df2['timestamp'] = df['timestamp']
df2.to_excel(distan_res_path)

# corr = df2['Mahalanobis'].corr(df2['accuracy'])
# print(corr)
# corr = df2['Mahalanobis'].corr(df2['accuracy'])
# print(corr)
corr = df2['dis'].corr(df2['accuracy'])
print(corr)
corr = df2['p'].corr(df2['accuracy'])
print(corr)
print(weights)

X = df2[keys]
clus = cluster.AgglomerativeClustering(n_clusters=None, linkage="average",affinity='l1',distance_threshold=0.5)
clus.fit(X)
label = np.reshape(clus.labels_, len(df2))
print(f"Number of points: {label.size}")
print(f"Number of clusters: {np.unique(label).size}")
print(label)
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
    cluster_dict['mean'] = samples['timestamp'].iloc[0]
    clusters.append(cluster_dict)
for clus_info in clusters:
    print(clus_info)

df_cluster = pd.DataFrame(clusters)
df_cluster = df_cluster.sort_values(by='num',ascending = False)
df_cluster['dis_to_normal'] = 0
mean_sample = df2[df2['timestamp'] == df_cluster['mean'].iloc[0]]
normal_label = mean_sample['label'].iloc[0]
print(mean_sample)
for i in range(len(df_cluster)):
    imid_sample = df2[df2['timestamp'] == df_cluster['mean'].iloc[i]]
    df_cluster['dis_to_normal'].iloc[i] = np.linalg.norm(np.array(imid_sample[keys].iloc[0])-np.array(mean_sample[keys].iloc[0]),ord=1)
print(df_cluster)
print(df_cluster['mean_acc'].corr(df_cluster['mean_noise']))
df_cluster.to_excel(cluster_res_path)


for i in range(len(df_cluster)):
    if df_cluster['num'].iloc[i] > 10:
        sub_label = df_cluster['label'].iloc[i]
        sub_df = df2.loc[df2['label']== sub_label] 
        sub_df.to_excel(root_cluster_res_path+ str(i)+'.xlsx')

# print(' normal data cluster, label =',normal_label)
# normal_df =  df2.loc[df2['label']== normal_label] 
# print(len(normal_df))

# X = normal_df[keys]
# from mlinsights.mlmodel import KMeansL1L2
# clus2 = KMeansL1L2(5, norm='L1')

# #cluster.AgglomerativeClustering(n_clusters=None, linkage="average",affinity='l1',distance_threshold=0.05)
# clus2.fit(X)
# label = np.reshape(clus2.labels_, len(normal_df))
# print(f"Number of points: {label.size}")
# print(f"Number of clusters: {np.unique(label).size}")
# print(label)

# clusters= []
# for i in range(np.unique(label).size):
#     cluster_dict = dict()
#     samples = normal_df.loc[label==i]
#     acc_avg =  samples['accuracy'].mean()
#     cluster_dict['label'] = i
#     cluster_dict['num'] = len(samples)
#     cluster_dict['mean_acc'] = acc_avg
#     cluster_dict['mean_noise'] = samples['dis'].mean()
#     cluster_dict['max_noise'] = samples['dis'].max()
#     mean_sample = samples[keys].mean(axis=0)
#     samples['in_class_dis'] = np.linalg.norm(samples[keys]-mean_sample,ord=1,axis=1)
#     samples = samples.sort_values(by='in_class_dis')
#     cluster_dict['mean'] = samples['timestamp'].iloc[0]
#     clusters.append(cluster_dict)
    
# for clus_info in clusters:
#     print(clus_info)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#     cmap=plt.cm.Paired,
#     aspect="auto",
#     origin="lower",
# )

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# # Plot the centroids as a white X
# plt.scatter(
#     centroids[:, 0],
#     centroids[:, 1],
#     marker="x",
#     s=169,
#     linewidths=3,
#     color="w",
#     zorder=10,
# )
# plt.title(
#     "K-means clustering on the digits dataset (PCA-reduced data)\n"
#     "Centroids are marked with white cross"
# )
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()