

from cmath import inf
from qiskit import *


# IBMQ.save_account('3eec804fe4d033aaf06c2566f8b4711c16fd802f10dde0075d1bf307340489ca5ea6045075c7c185272f2a9566a7a1630a9cd2c60d52c6e6dbd901438ee2e4c5', overwrite=True)
# GMU_hub = 'ibm-q-education'  # 'ibm-q'
# print("load the account successfully!")



import qiskit.providers
from qiskit import IBMQ
import datetime
import yaml
import argparse
import pandas as pd
import numpy as np
import os
from qiskit.providers.models import BackendProperties
from my_property import IBMMachinePropertiesFactory,MyBackendNoise
import pickle
import sys


provider = IBMQ.load_account()
property_factory = IBMMachinePropertiesFactory(provider)

backend_name = 'ibm_perth'
fname = 'calibration_data/real_device/{}-{}.csv'.format(backend_name, datetime.datetime.now().strftime("%Y-%m-%d"))

query_date = datetime.datetime(2022,11,20,21,54,0)


config, prop, backend = property_factory.query_backend_info(backend_name, None)
prop_dict = prop.to_dict()
# open file for writing, "w" 
# f = open('history_property/'+query_date.strftime("%Y-%m-%d-%H-%M") + ".pkl","wb")
# pickle.dump(prop_dict,f)
# f.close()
backend_property = MyBackendNoise(backend_name,query_date)
backend_property.set_property(prop)
noise =  backend_property.get_noise()
date_format =  "%Y-%m-%d-%H-%M"

df = pd.DataFrame.from_dict(noise)
print(noise['timestamp'])
# df.to_csv(fname)
if os.path.isfile(fname):
    df.to_csv(fname, mode = 'a', index = False, header = False)
else:
    df.to_csv(fname, mode = 'w', index = False, header = True)
