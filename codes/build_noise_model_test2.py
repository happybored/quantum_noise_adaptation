from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.device.models import (basic_device_readout_errors,
                                                      basic_device_gate_errors)
from qiskit.providers.aer.noise.device.parameters import (thermal_relaxation_values,gate_param_values)
from qiskit.providers.models import BackendProperties
import sys
from my_property import MyBackendNoise,IBMMachinePropertiesFactory
import copy

import warnings
warnings.filterwarnings("ignore")


provider = IBMQ.load_account()
factory =  IBMMachinePropertiesFactory(provider)
backend_name = 'ibmq_belem'
config,properties,backend =  factory.query_backend_info(backend_name)

my_backend_noise=MyBackendNoise(backend_name)
my_backend_noise.set_property(properties)
noise_config = my_backend_noise.get_noise()
print('before change:')
print(noise_config)
noise_config2 = copy.deepcopy(noise_config)
noise_config2['id0'] = 0.05
noise_config2['sx0'] = 0.05
noise_config2['x0'] = 0.05
noise_config2['cx0_1'] = 0.1
noise_config2['cx1_0'] = 0.1
noise_config2['prob_meas0_prep1_q0'] = 0.09
noise_config2['prob_meas1_prep0_q0'] = 0.0136
noise_config2['readout_error_q0'] = 0.01518

my_backend_noise.set_noise(noise_config2)
properties2 = my_backend_noise.get_property()
noise_config3 = my_backend_noise.get_noise()
print('---'*10)
print('after change:')
print(noise_config3)
print('==='*20)

print('before change:')
property_dict = properties.to_dict()
for qubit in property_dict['qubits']:
    for info in qubit:
            if info['name'] in ['T1','T2']:
                    print(info['name'] ,info['value'])
print('---'*10)
print('after change:')
property_dict = properties2.to_dict()
for qubit in property_dict['qubits']:
    for info in qubit:
            if info['name'] in ['T1','T2']:
                    print(info['name'] ,info['value'])

print('==='*20)
print(properties==properties2)

# #build 2 noise model
my_noise_model = NoiseModel(basis_gates=config.basis_gates)
print('before change:')
for qubits, error in basic_device_readout_errors(properties):
    my_noise_model.add_readout_error(error, qubits)
    print('readout error prob:',error.probabilities,' qubits:',qubits)
    break
gate_errors = basic_device_gate_errors(properties,temperature=20)
for name, qubits, error in gate_errors:
    my_noise_model.add_quantum_error(error, name, qubits)
    print('gate error:',error.probabilities,' name:',name,' qubits:',qubits)
    # break

print('---'*10)
print('after change:')

my_noise_model2 = NoiseModel(basis_gates=config.basis_gates)

for qubits, error in basic_device_readout_errors(properties2):
    my_noise_model2.add_readout_error(error, qubits)
    print('readout error prob:',error.probabilities,' qubits:',qubits)
    break
gate_errors = basic_device_gate_errors(properties2,temperature=20)
for name, qubits, error in gate_errors:
    my_noise_model2.add_quantum_error(error, name, qubits)
    print('gate error:',error.probabilities,' name:',name,' qubits:',qubits)
    # break