import numpy
import pandas as pd

path ='calibration_data/cluster_res_eq/ibmq_belem-tech-offline-online-cluster-database.xlsx'
path2 = 'calibration_data/ibmq_belem-tech.csv'

df2 = pd.read_excel(path)
df  = pd.read_csv(path2)
df3 = df.loc[df.timestamp.isin(df2.timestamp)]
print(df3)
# df3['lr'] = df2['lr']

df3.to_excel(path)