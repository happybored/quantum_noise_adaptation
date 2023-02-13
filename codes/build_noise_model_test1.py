from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.device.models import (basic_device_readout_errors,
                                                      basic_device_gate_errors)
from qiskit.providers.aer.noise.device.parameters import (thermal_relaxation_values,gate_param_values)
from qiskit.providers.models import BackendProperties
import sys
from my_property import MyBackendNoise,IBMMachinePropertiesFactory


IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_belem')
properties = backend.properties()
properties_dict = properties.to_dict()
my_properties = BackendProperties.from_dict(properties_dict)
print(my_properties == properties)





def build_noise_model_from_properties_and_config(properties,basis_gates):
    my_noise_model = NoiseModel(basis_gates=basis_gates)
    

    for qubits, error in basic_device_readout_errors(properties):
        my_noise_model.add_readout_error(error, qubits)

    gate_errors = basic_device_gate_errors(properties)
    for name, qubits, error in gate_errors:
        my_noise_model.add_quantum_error(error, name, qubits)
    return my_noise_model



backend_noise_model = NoiseModel.from_backend(backend)
config = backend.configuration()
my_noise_model = NoiseModel(basis_gates=config.basis_gates)
readout_errors = basic_device_readout_errors(properties)

for qubits, error in basic_device_readout_errors(properties):
    my_noise_model.add_readout_error(error, qubits)
    # print('error prob:',error.probabilities,' \n qubits:',qubits)

gate_errors = basic_device_gate_errors(properties)
for name, qubits, error in gate_errors:
    my_noise_model.add_quantum_error(error, name, qubits)
    # print('error:',error.probabilities,' name:',name,' qubits:',qubits)

print(backend_noise_model == my_noise_model)
