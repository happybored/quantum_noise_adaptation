from tkinter.messagebox import NO
from turtle import back
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
from .qiskit_macros import IBMQ_NAMES
from tqdm import tqdm
from torchpack.utils.logging import logger
from qiskit.transpiler import PassManager
import numpy as np
import datetime
import sys


class EmptyPassManager(PassManager):
    def run(
        self,
        circuits,
        output_name: str = None,
        callback=None
    ):
        return circuits


def run_job_worker(data):
    while True:
        try:
            job = execute(**(data[0]))
            qiskit_verbose = data[1]
            if qiskit_verbose:
                job_monitor(job, interval=1)
            result = job.result()
            counts = result.get_counts()
            circ_num = len(data[0]['parameter_binds'])
            logger.info(f'run job worker successful, circuit number = {circ_num}')
            break
        except Exception as e:
            if "Job was cancelled" in str(e):
                logger.warning(f"Job is cancelled manually.")
                return None
            else:
                logger.warning(f"Job failed because {e}, rerun now.")
    

    return counts

from qiskit import IBMQ

# IBMQ.save_account(TOKEN)
# IBMQ.load_account() # Load account from disk
# provider = IBMQ.get_provider(hub='ibm-q-research-2', group='george-mason-uni-1', project='main')
# provider = IBMQ.get_provider(hub='ibm-q-education', group='george-mason-uni-1', project='hardware-acceler')
class QiskitProcessor(object):
    def __init__(self,
                 use_real_qc=False,
                 backend_name=None,
                 backend = None,
                 noise_model_name=None,
                 noise_model = None,
                 coupling_map_name=None,
                 coupling_map = None,
                 basis_gates_name=None,
                 basis_gates =None,
                 n_shots=8192,
                 initial_layout=None,
                 seed_transpiler=42,
                 seed_simulator=42,
                 optimization_level=1,
                 max_jobs=1,
                 max_experiments = 64,
                 remove_ops=False,
                 remove_ops_thres=1e-4,
                 transpile_with_ancilla=True,
                 hub='ibm-q-education',
                 group='george-mason-uni-1',
                 project='hardware-acceler',
                 layout_method=None,
                 routing_method=None
                 ):
        self.use_real_qc = use_real_qc
        self.noise_model_name = noise_model_name
        self.backend_name = backend_name
        self.coupling_map_name = coupling_map_name
        self.basis_gates_name = basis_gates_name
        self.n_shots = n_shots
        self.initial_layout = initial_layout
        self.seed_transpiler = seed_transpiler
        self.seed_simulator = seed_simulator
        self.optimization_level = optimization_level
        self.max_jobs = max_jobs
        self.max_experiments = max_experiments
        self.transpile_with_ancilla = transpile_with_ancilla

        self.layout_method = layout_method
        self.routing_method = routing_method

        self.hub = hub
        self.group = group
        self.project = project
        self.backend = backend
        self.provider = None
        self.noise_model = noise_model
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates
        self.properties = None
        self.empty_pass_manager = EmptyPassManager()

        self.transpiled_circs = None

        self.remove_ops = remove_ops
        self.remove_ops_thres = remove_ops_thres

        self.qiskit_init()

    def get_noise_model(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            self.properties = backend.properties()
            noise_model = NoiseModel.from_backend(backend)
        else:
            noise_model = None

        return noise_model

    def get_coupling_map(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            coupling_map = backend.configuration().coupling_map
        else:
            if name == 'four_all':
                coupling_map = [[0, 1], [1, 0],
                                [0, 2], [2, 0],
                                [0, 3], [3, 0],
                                [1, 2], [2, 1],
                                [1, 3], [3, 1],
                                [2, 3], [3, 2]]
            else:
                coupling_map = None

        return coupling_map

    def get_basis_gates(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            basis_gates = backend.configuration().basis_gates
        else:
            basis_gates = None

        return basis_gates

    def qiskit_init(self):
        self.provider = None
        self.properties = None


        if self.backend ==None:
            if self.use_real_qc:
                # IBMQ.load_account()
                self.provider = get_provider_hub_group_project(
                    hub=self.hub,
                    group=self.group,
                    project=self.project,
                )
                self.backend = self.provider.get_backend(self.backend_name)
                self.properties = self.backend.properties()
                self.coupling_map = self.get_coupling_map(self.backend_name)
                
            else:
                # use simulator
                self.backend = Aer.get_backend('qasm_simulator',
                                               max_parallel_experiments=0)
                # IBMQ.load_account()
                self.provider = get_provider_hub_group_project(
                    hub=self.hub,
                    group=self.group,
                    project=self.project,
                )
                if self.noise_model ==None:
                    self.noise_model = self.get_noise_model(self.noise_model_name)
                    
                if self.coupling_map ==None and self.coupling_map_name ==None:
                    self.coupling_map = self.get_coupling_map(self.noise_model_name)
                elif self.coupling_map ==None:
                    self.coupling_map = self.get_coupling_map(self.coupling_map_name)
                if self.basis_gates == None and self.basis_gates_name ==None:
                    self.basis_gates = self.get_basis_gates(self.noise_model_name)
                elif self.basis_gates == None:
                    self.basis_gates = self.get_basis_gates(self.basis_gates_name)
        else:
            self.backend_name = self.backend.name()
            if self.coupling_map == None:
                self.coupling_map = self.backend.configuration().coupling_map
            if self.basis_gates == None:
                self.basis_gates = self.backend.configuration().basis_gates

    def set_layout(self, layout):
        self.initial_layout = layout
    
    def set_backend(self, backend):
        self.backend = backend

    def transpile(self, circs):
        if not self.transpile_with_ancilla and self.coupling_map is not None:
            # only use same number of physical qubits as virtual qubits
            # !! the risk is that the remaining graph is not a connected graph,
            # need fix this later
            coupling_map = []
            for pair in self.coupling_map:
                if all([p_wire < len(circs.qubits) for p_wire in pair]):
                    coupling_map.append(pair)
        else:
            coupling_map = self.coupling_map
        transpiled_circs = transpile(circuits=circs,
                                     backend=self.backend,
                                     basis_gates=self.basis_gates,
                                     coupling_map=coupling_map,
                                     initial_layout=self.initial_layout,
                                     seed_transpiler=self.seed_transpiler,
                                     optimization_level=self.optimization_level
                                     )
        return transpiled_circs

    def preprocess_parameterized(self,
                                 q_device,
                                 q_layer_parameterized,
                                 q_layer_fixed,
                                 q_layer_measure,
                                 x,
                                 ):
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list)
        circ_fixed = tq2qiskit(q_device, q_layer_fixed,
                               remove_ops=self.remove_ops,
                               remove_ops_thres=self.remove_ops_thres)

        circ_measurement = tq2qiskit_measurement(q_device, q_layer_measure)
        circ = circ_parameterized + circ_fixed + circ_measurement

        # logger.info(f'Before transpile: {get_circ_stats(circ)}')
        transpiled_circ = self.transpile(circ)
        # logger.info(f'After transpile: {get_circ_stats(transpiled_circ)}')
        self.transpiled_circs = [transpiled_circ]
        # construct the parameter_binds
        binds_all = []
        for inputs_single in x:
            binds = {}
            for k, input_single in enumerate(inputs_single):
                binds[params[k]] = input_single.item()
            binds_all.append(binds)

        return transpiled_circ, binds_all

    def process_parameterized(self, q_device: tq.QuantumDevice,
                              q_layer_parameterized: tq.QuantumModule,
                              q_layer_fixed: tq.QuantumModule,
                              q_layer_measure: tq.QuantumModule,
                              x,
                              parallel=True):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required.
        q_layer_parameterized needs to have a func_list to specify the gates
        for parallel:
        JobManager has bugs when submitting job, so use multiprocessing instead
        """
        transpiled_circ, binds_all = self.preprocess_parameterized(
            q_device, q_layer_parameterized, q_layer_fixed,
            q_layer_measure, x)

        if parallel:
            if self.use_real_qc:
                if hasattr(self.backend.configuration(), 'max_experiments'):
                    chunk_size = self.backend.configuration().max_experiments
                else:
                    # using simulator, apply multithreading
                    chunk_size = len(binds_all) // self.max_jobs
            else:
                chunk_size = min(self.max_experiments,len(binds_all))

            if chunk_size == 0:
                split_binds = [binds_all]
            else:
                split_binds = [binds_all[i:i + chunk_size] for i in range(
                    0, len(binds_all), chunk_size)]
            
            qiskit_verbose = False
            feed_dicts = []
            # print(self.backend)
            # print(self.noise_model)
            for split_bind in split_binds:
                feed_dict = {
                    'experiments': transpiled_circ,
                    'backend': self.backend,
                    'pass_manager': self.empty_pass_manager,
                    'shots': self.n_shots,
                    'seed_simulator': self.seed_simulator,
                    'noise_model': self.noise_model,
                    'parameter_binds': split_bind,
                }
                feed_dicts.append([feed_dict, qiskit_verbose])

            p = multiprocessing.Pool(self.max_jobs)
            results = p.map(run_job_worker, feed_dicts)
            p.close()

            if all(isinstance(result, dict) for result in results):
                counts = results
            else:
                if isinstance(results[-1], dict):
                    results[-1] = [results[-1]]
                counts = list(itertools.chain(*results))
        else:
            job = execute(experiments=transpiled_circ,
                          backend=self.backend,
                          pass_manager=self.empty_pass_manager,
                          shots=self.n_shots,
                          seed_simulator=self.seed_simulator,
                          noise_model=self.noise_model,
                          parameter_binds=binds_all
                          )
            job_monitor(job, interval=1)

            result = job.result()
            counts = result.get_counts()

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)
        return measured_qiskit

    def preprocess_parameterized_and_shift(self,
                                 q_device,
                                 q_layer_parameterized,
                                 q_layer_fixed,
                                 q_layer_measure,
                                 x,
                                 shift_encoder,
                                 shift_this_step):
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list)
        circ_fixed_list = []
        circ_fixed = tq2qiskit(q_device, q_layer_fixed,
                               remove_ops=self.remove_ops,
                               remove_ops_thres=self.remove_ops_thres)
        circ_fixed_list.append(circ_fixed)

        # not shift encoder ==> shift fixed layer
        if not shift_encoder:
            for i, named_param in enumerate(q_layer_fixed.named_parameters()):
                if shift_this_step[i]:
                    param = named_param[-1]
                    param.copy_(param + np.pi*0.5)
                    circ_fixed = tq2qiskit(q_device, q_layer_fixed, remove_ops=self.remove_ops, remove_ops_thres=self.remove_ops_thres)
                    circ_fixed_list.append(circ_fixed)
                    param.copy_(param - np.pi)
                    circ_fixed = tq2qiskit(q_device, q_layer_fixed, remove_ops=self.remove_ops, remove_ops_thres=self.remove_ops_thres)
                    circ_fixed_list.append(circ_fixed)
                    param.copy_(param + np.pi*0.5)
        
        self.transpiled_circs = []
        for circ_fixed in circ_fixed_list:
            circ = circ_parameterized + circ_fixed
            v_c_reg_mapping = q_layer_measure.v_c_reg_mapping
            if v_c_reg_mapping is not None:
                for q_reg, c_reg in v_c_reg_mapping['v2c'].items():
                    circ.measure(q_reg, c_reg)
            else:
                circ.measure(list(range(q_device.n_wires)), list(range(
                    q_device.n_wires)))

            transpiled_circ = self.transpile(circ)
            self.transpiled_circs.append(transpiled_circ)
        # construct the parameter_binds
        binds_all = []
        if shift_encoder:
            for idx in range(x.size()[1]):
                x[:, idx] += np.pi * 0.5
                for inputs_single in x:
                    binds = {}
                    for k, input_single in enumerate(inputs_single):
                        binds[params[k]] = input_single.item()
                    binds_all.append(binds)
                
                x[:, idx] -= np.pi
                for inputs_single in x:
                    binds = {}
                    for k, input_single in enumerate(inputs_single):
                        binds[params[k]] = input_single.item()
                    binds_all.append(binds)
                
                x[:, idx] += np.pi * 0.5
        else:
            for inputs_single in x:
                binds = {}
                for k, input_single in enumerate(inputs_single):
                    binds[params[k]] = input_single.item()
                binds_all.append(binds)


        return self.transpiled_circs, binds_all


    def process_parameterized_and_shift(self, q_device: tq.QuantumDevice,
                              q_layer_parameterized: tq.QuantumModule,
                              q_layer_fixed: tq.QuantumModule,
                              q_layer_measure: tq.QuantumModule,
                              x,
                              shift_encoder=False,
                              parallel=True,
                              shift_this_step=None):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required.
        q_layer_parameterized needs to have a func_list to specify the gates
        for parallel:
        JobManager has bugs when submitting job, so use multiprocessing instead
        """
        transpiled_circs, binds_all = self.preprocess_parameterized_and_shift(
            q_device, q_layer_parameterized, q_layer_fixed,
            q_layer_measure, x, shift_encoder, shift_this_step)
        
        time_spent_list = []

        if parallel:
            if hasattr(self.backend.configuration(), 'max_experiments'):
                chunk_size = self.backend.configuration().max_experiments
            else:
                # using simulator, apply multithreading
                chunk_size = len(binds_all) // self.max_jobs

            split_binds = [binds_all[i:i + chunk_size] for i in range(
                0, len(binds_all), chunk_size)]

            qiskit_verbose = self.max_jobs <= 6
            feed_dicts = []
            for split_bind in split_binds:
                feed_dict = {
                    'experiments': transpiled_circs,
                    'backend': self.backend,
                    'pass_manager': self.empty_pass_manager,
                    'shots': self.n_shots,
                    'seed_simulator': self.seed_simulator,
                    'noise_model': self.noise_model,
                    'parameter_binds': split_bind,
                }
                feed_dicts.append([feed_dict, qiskit_verbose])

            p = multiprocessing.Pool(self.max_jobs)
            results = p.map(run_job_worker, feed_dicts)
            p.close()

            if all(isinstance(result, dict) for result in results):
                counts = results
            else:
                if isinstance(results[-1], dict):
                    results[-1] = [results[-1]]
                counts = list(itertools.chain(*results))
        else:
            chunk_size = 75 // len(binds_all)
            split_circs = [transpiled_circs[i:i + chunk_size] for i in range(0, len(transpiled_circs), chunk_size)]
            counts = []
            total_time_spent = datetime.timedelta()
            total_cont = 0
            for circ in split_circs:
                while True:
                    try:
                        job = execute(experiments=circ,
                                    backend=self.backend,
                                    pass_manager=self.empty_pass_manager,
                                    shots=self.n_shots,
                                    seed_simulator=self.seed_simulator,
                                    noise_model=self.noise_model,
                                    parameter_binds=binds_all
                                    )
                        # job_monitor(job, interval=1)
                        result = job.result()#qiskit.providers.ibmq.job.exceptions.IBMQJobFailureError:Job has failed. Use the error_message() method to get more details
                        counts = counts + result.get_counts()
                        # time_per_step = job.time_per_step()
                        # time_spent = time_per_step['COMPLETED'] - time_per_step['RUNNING'] + time_per_step['QUEUED'] - job.time_per_step()['CREATING']
                        # time_spent_list.append(time_spent)
                        # print(time_spent)
                        # total_time_spent += time_spent
                        # total_cont += 1
                        # print(total_time_spent / total_cont)
                        break
                    except (QiskitError) as e:
                        logger.warning('Job failed, rerun now.')
                        print(e.message)

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit, time_spent_list


    def process_multi_measure(self,
                              q_device: tq.QuantumDevice,
                              q_layer: tq.QuantumModule,
                              q_layer_measure: tq.QuantumModule,):
        obs_list = q_layer_measure.obs_list
        circ_fixed = tq2qiskit(q_device, q_layer,
                               remove_ops=self.remove_ops,
                               remove_ops_thres=self.remove_ops_thres)

        transpiled_circ_fixed = self.transpile(circ_fixed)

        circ_all = []

        for hamil in obs_list:
            circ_diagonalize = QuantumCircuit(q_device.n_wires,
                                              q_device.n_wires)

            # diagonalize the measurements
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'x':
                    circ_diagonalize.h(qubit=wire)
                elif observable == 'y':
                    circ_diagonalize.z(qubit=wire)
                    circ_diagonalize.s(qubit=wire)
                    circ_diagonalize.h(qubit=wire)

            circ_measurement = tq2qiskit_measurement(q_device, q_layer_measure)

            circ_diagonalize = circ_diagonalize + circ_measurement

            transpiled_circ_diagonalize = self.transpile(circ_diagonalize)
            circ_all.append(transpiled_circ_fixed +
                            transpiled_circ_diagonalize)

        self.transpiled_circs = circ_all

        if hasattr(self.backend.configuration(), 'max_experiments'):
            chunk_size = self.backend.configuration().max_experiments
        else:
            # using simulator, apply multithreading
            chunk_size = len(circ_all) // self.max_jobs

        split_circs = [circ_all[i:i + chunk_size] for i in range(
            0, len(circ_all), chunk_size)]

        qiskit_verbose = self.max_jobs <= 2
        feed_dicts = []
        for split_circ in split_circs:
            feed_dict = {
                'experiments': split_circ,
                'backend': self.backend,
                'pass_manager': self.empty_pass_manager,
                'shots': self.n_shots,
                'seed_simulator': self.seed_simulator,
                'noise_model': self.noise_model,
            }
            feed_dicts.append([feed_dict, qiskit_verbose])

        p = multiprocessing.Pool(self.max_jobs)
        results = p.map(run_job_worker, feed_dicts)
        p.close()

        if all(isinstance(result, dict) for result in results):
            counts = results
        else:
            if isinstance(results[-1], dict):
                results[-1] = [results[-1]]
            counts = list(itertools.chain(*results))

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)

        measured_qiskit = torch.tensor(measured_qiskit,
                                       device=q_device.state.device)

        return measured_qiskit

    def process(self, q_device: tq.QuantumDevice, q_layer: tq.QuantumModule,
                q_layer_measure: tq.QuantumModule, x):
        circs = []
        for i, x_single in tqdm(enumerate(x)):
            circ = tq2qiskit(q_device, q_layer, x_single.unsqueeze(0))
            circ_measurement = tq2qiskit_measurement(q_device, q_layer_measure)
            circ = circ + circ_measurement

            circs.append(circ)

        transpiled_circs = self.transpile(circs)
        self.transpiled_circs = transpiled_circs

        job = execute(experiments=transpiled_circs,
                      backend=self.backend,
                      shots=self.n_shots,
                      initial_layout=self.initial_layout,
                      seed_transpiler=self.seed_transpiler,
                      seed_simulator=self.seed_simulator,
                      coupling_map=self.coupling_map,
                      basis_gates=self.basis_gates,
                      noise_model=self.noise_model,
                      optimization_level=self.optimization_level,
                      )
        job_monitor(job, interval=1)

        result = job.result()
        counts = result.get_counts()
        print(counts)
        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit

    def process_ready_circs(self, q_device, circs_all, parallel=True):
        circs_all_transpiled = []
        for circ in tqdm(circs_all):
            circs_all_transpiled.append(self.transpile(circ))

        circs_all = circs_all_transpiled

        if parallel:
            if hasattr(self.backend.configuration(), 'max_experiments'):
                chunk_size = self.backend.configuration().max_experiments
            else:
                # using simulator, apply multithreading
                chunk_size = len(circs_all) // self.max_jobs
                if chunk_size == 0:
                    chunk_size = 1

            split_circs = [circs_all[i:i + chunk_size] for i in range(
                0, len(circs_all), chunk_size)]

            qiskit_verbose = self.max_jobs <= 6
            feed_dicts = []
            for split_circ in split_circs:
                feed_dict = {
                    'experiments': split_circ,
                    'backend': self.backend,
                    'pass_manager': self.empty_pass_manager,
                    'shots': self.n_shots,
                    'seed_simulator': self.seed_simulator,
                    'noise_model': self.noise_model,
                }
                feed_dicts.append([feed_dict, qiskit_verbose])

            p = multiprocessing.Pool(self.max_jobs)
            results = p.map(run_job_worker, feed_dicts)
            p.close()

            if all(isinstance(result, dict) for result in results):
                counts = results
            else:
                if isinstance(results[-1], dict):
                    results[-1] = [results[-1]]
                counts = list(itertools.chain(*results))
        else:
            job = execute(experiments=circs_all,
                          backend=self.backend,
                          pass_manager=self.empty_pass_manager,
                          shots=self.n_shots,
                          seed_simulator=self.seed_simulator,
                          noise_model=self.noise_model,
                          )
            job_monitor(job, interval=1)

            result = job.result()
            counts = result.get_counts()

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        
        measured_torch = torch.tensor(measured_qiskit)

        return measured_torch