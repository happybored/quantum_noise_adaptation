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

path ='/home/zhirui/Software/quantum/adaptive_noise/calibration_data/1/ibmq_belem-tech-online-error-detector.xlsx'
df = pd.read_excel(path)
df =df[['error_flag','dynamic_training']]
df = df.sort_values(by='dynamic_training')
print(df)
X = []
Z = []
total_num = len(df)
for i in range(5,total_num-5,2):
    positive_number = i
    negitive_number = total_num -i
    true_positive = df['error_flag'].iloc[0:i].values.sum()
    false_negtive = positive_number -true_positive
    false_positive = df['error_flag'].iloc[i:total_num].values.sum()
    true_negtive = negitive_number - false_positive
    TPR = true_positive *1.0 /positive_number
    FPR = false_positive *1.0 /negitive_number
    TR = (true_positive+true_negtive)/total_num
    print(true_positive)
    print(false_positive)
    # sys.exit(0)
    
    # print()
    X.append([true_positive,false_negtive,false_positive,true_negtive])
    Z.append(TR)

index = Z.index(max(Z)) 
print(max(Z))
print(X[index])
# plt.plot(X,Y)
# plt.show()
