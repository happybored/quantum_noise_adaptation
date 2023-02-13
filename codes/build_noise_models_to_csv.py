from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.device.models import (basic_device_readout_errors,
                                                      basic_device_gate_errors)
from qiskit.providers.aer.noise.device.parameters import (thermal_relaxation_values,gate_param_values)
from qiskit.providers.models import BackendProperties
import sys
from my_property import MyBackendNoise,IBMMachinePropertiesFactory
import copy
import pandas as pd
import warnings
import os
import datetime
warnings.filterwarnings("ignore")


provider = IBMQ.load_account()
factory =  IBMMachinePropertiesFactory(provider)
backend_name = 'ibmq_lima' #backend name
config,properties,backend =  factory.query_backend_info(backend_name)
my_backend_noise=MyBackendNoise(backend_name)
my_backend_noise.set_property(properties)
noise_config = my_backend_noise.get_noise()
print(noise_config)
coupling_map = config.coupling_map
print(coupling_map)
fname = 'calibration_data/experiment-data-{}.csv'.format(backend_name) #datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

gate_error = [0.01,0.01,0.01,0.01,0.01]
prob_meas0_prep1 = [0.09,0.09,0.09,0.09,0.09]
prob_meas1_prep0 = [0.0136,0.0136,0.0136,0.0136,0.0136]
def generate_noise_config(gate_error,prob_meas0_prep1,prob_meas1_prep0,noise_config):
    noise_config2 = copy.deepcopy(noise_config)
    for i in range(5):
        noise_config2['id'+str(i)] = gate_error[i]
        noise_config2['sx'+str(i)] = gate_error[i]
        noise_config2['x'+ str(i)] = gate_error[i]
        noise_config2['prob_meas0_prep1_q'+str(i)] = prob_meas0_prep1[i]
        noise_config2['prob_meas1_prep0_q'+str(i)] = prob_meas1_prep0[i]
        noise_config2['readout_error_q'+str(i)] = (prob_meas0_prep1[i]+prob_meas1_prep0[i])/2
    for pair in coupling_map:
        noise_config2['cx{}_{}'.format(pair[0],pair[1])] = 1- (1-gate_error[pair[0]])*(1-gate_error[pair[1]])
    return noise_config2
def save(noise_config):
    df = pd.DataFrame.from_dict(noise_config)
    if os.path.isfile(fname):
        df.to_csv(fname, mode = 'a', index = False, header = False)
    else:
        df.to_csv(fname, mode = 'w', index = False, header = True)

noise_config2 = generate_noise_config(gate_error,prob_meas0_prep1,prob_meas1_prep0,noise_config)
save(noise_config2)

shifts = [0.6,0.8,1,1.2,1.4] #noise shift scale 
for s1 in shifts:
    for s2 in shifts:
        gate_error = [0.01,0.01,0.01,0.01,0.01]
        prob_meas0_prep1 = [0.09,0.09,0.09,0.09,0.09]
        prob_meas1_prep0 = [0.0136,0.0136,0.0136,0.0136,0.0136]
        gate_error[0] = gate_error[0]*s1
        gate_error[1] = gate_error[1]*s1
        gate_error[2] = gate_error[2]*s2
        gate_error[3] = gate_error[3]*s2
        prob_meas0_prep1[0] = prob_meas0_prep1[0]*s1
        prob_meas0_prep1[1] = prob_meas0_prep1[1]*s1
        prob_meas0_prep1[2] = prob_meas0_prep1[2]*s2
        prob_meas0_prep1[3] = prob_meas0_prep1[3]*s2
        prob_meas1_prep0[0] = prob_meas1_prep0[0]*s1
        prob_meas1_prep0[1] = prob_meas1_prep0[1]*s1
        prob_meas1_prep0[2] = prob_meas1_prep0[2]*s2
        prob_meas1_prep0[3] = prob_meas1_prep0[3]*s2
        noise_config2 = generate_noise_config(gate_error,prob_meas0_prep1,prob_meas1_prep0,noise_config)
        save(noise_config2)






