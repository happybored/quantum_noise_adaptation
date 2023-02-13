from qiskit import IBMQ
import datetime

provider = IBMQ.load_account()
backend = provider.backend.ibmq_belem

properties = backend.properties(datetime=datetime.datetime(2022,1,1)) # change the date
properties_dict = properties.to_dict()
print
()
for key,val in properties_dict.items():
    print(key,':')
    print(val)
    print()
print('='*50)
qubits = properties_dict['qubits']
for qubit in qubits:
    print(qubit)
    print()

print('='*50)
gates = properties_dict['gates']
for gate in gates:
    print(gate)
    print()



class SimpleProperty(object):
    def __init__(self,properties_dict):
        self.backend_name =  properties_dict['backend_name']

quibits_properities = properties_dict['qubits']
print(len(quibits_properities))

# def quantum_properties_parser():

