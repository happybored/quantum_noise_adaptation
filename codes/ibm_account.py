from qiskit import IBMQ


IBMQ.load_account() # Load account from disk
# provider = IBMQ.get_provider(hub='ibm-q-education', group='george-mason-uni-1', project='hardware-acceler')
# provider = IBMQ.get_provider(hub='ibm-q-research-2', group='george-mason-uni-1', project='main')
provider = IBMQ.get_provider(hub='ibm-q-lanl', group='lanl', project='quantum-optimiza')

