from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit import IBMQ
import pandas as pd
import os
result = dict()
result['acc'] = [1.2]
result['fff'] = [2.3]
result['ggg'] = [2.4]

df3 = pd.read_csv('calibration_data/prune_ratios.xlsx' )
df3 = df3[['timestamp','pr']]
prune_ratio = dict()
for i in range(len(df3)):
    key = df3['timestamp'].loc[i].split(' ')[0]
    prune_ratio[key] = df3['pr'].loc[i]

print(prune_ratio)