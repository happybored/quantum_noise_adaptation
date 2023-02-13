from turtle import back
import qiskit
import torch
import torchquantum as tq
import pathos.multiprocessing as multiprocessing
import itertools

from qiskit import Aer, execute, IBMQ, transpile, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.exceptions import QiskitError
from torchquantum.plugins import tq2qiskit, tq2qiskit_parameterized, \
    tq2qiskit_measurement
from torchquantum.utils import (get_expectations_from_counts, get_provider,
                                get_provider_hub_group_project,
                                get_circ_stats)
from qiskit_macros import IBMQ_NAMES
from tqdm import tqdm
from torchpack.utils.logging import logger
from qiskit.transpiler import PassManager
from qiskit.providers.fake_provider import FakeProvider
from qiskit.providers.basicaer import BasicAerProvider
import numpy as np
import datetime

fakeProvider = FakeProvider()
# backend = fakeProvider.get_backend(name = 'fake_valencia')
baskicAerProvider = BasicAerProvider()
backend = baskicAerProvider.get_backend('qasm_simulator')
config = backend.configuration()
print(config.to_dict()['backend_name'])
name = backend.name()
print(name)