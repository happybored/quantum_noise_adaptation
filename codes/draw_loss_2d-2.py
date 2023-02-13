from torchpack.datasets.dataset import Dataset
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchquantum.plugins import tq2qiskit, qiskit2tq
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.layers import SethLayer0

from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from modified.qiskit_processor import QiskitProcessor
from my_property import IBMMachinePropertiesFactory,MyBackendNoise
from qiskit import IBMQ
import datetime
import warnings
from modified.noise_model import NoiseModelTQ
from qiskit.providers.aer.noise.device.models import (basic_device_readout_errors,
                                                      basic_device_gate_errors)
from qiskit.providers.aer.noise import NoiseModel
import math
from torchquantum.utils import (build_module_from_op_list,
                                build_module_op_list,
                                get_v_c_reg_mapping,
                                get_p_c_reg_mapping,
                                get_p_v_reg_mapping,
                                get_cared_configs)
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import interpolate
# change qiskit processor to fake/ zidingyi backend
warnings.filterwarnings("ignore")

class Classification2Dataset(torch.utils.data.Dataset):
    def __init__(self, num=11):
        self.data = []
        self.target = []
        sum0 = 0
        sum1 = 0
        for x in np.linspace(0, 1, num=num):
            for y in np.linspace(0, 1, num=num):
                self.data.append(torch.tensor([x, y]))
                if (x**2 + y**2 <= 0.55**2 or (x-1)**2 + (y-1)**2 <= 0.55**2):
                    self.target.append(1)
                    sum1 = sum1 + 1
                else:
                    self.target.append(0)
                    sum0 = sum0 + 1
            # print(self.target[-num:])

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'target': self.target[idx]}

    def __len__(self):
        return len(self.target) - 1

class Simple2Class(Dataset):
    def __init__(self):
        train_dataset = Classification2Dataset()
        valid_dataset = Classification2Dataset(num=10)
        datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': valid_dataset}
        super().__init__(datasets)

class Q2Model(tq.QuantumModule):
    class Ansatz(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.op0 = tq.CNOT(has_params=False, trainable=False)
            self.n_wires = 2
            self.op1 = tq.RY(has_params=True, trainable=True)
            self.op2 = tq.RY(has_params=True, trainable=True)
            # self.op3 = tq.CNOT(has_params=False, trainable=False)

        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.op1(self.q_device, wires=0)
            self.op0(self.q_device, wires=[0, 1])
            self.op1(self.q_device, wires=0)
            self.op0(self.q_device, wires=[0, 1])
            
            self.op2(self.q_device, wires=1)
            self.op0(self.q_device, wires=[1, 0])
            self.op2(self.q_device, wires=1)
            self.op0(self.q_device, wires=[1, 0])

                       
            # self.op0(self.q_device, wires=[1, 0])

            # self.op1(self.q_device, wires=0)
            # self.op2(self.q_device, wires=1)
            # self.op0(self.q_device, wires=[0, 1])
            # self.op1(self.q_device, wires=1)
            # self.op0(self.q_device, wires=[1, 0])

            # self.op1(self.q_device, wires=0)
            # self.op2(self.q_device, wires=1)
            # self.op0(self.q_device, wires=[0, 1])
            # self.op1(self.q_device, wires=1)
            # self.op0(self.q_device, wires=[1, 0])

            # self.op1(self.q_device, wires=0)
            # self.op2(self.q_device, wires=1)
            # self.op0(self.q_device, wires=[0, 1])
            # self.op1(self.q_device, wires=1)
            # self.op0(self.q_device, wires=[1, 0])        
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([{'input_idx': [0], 'func': 'ry', 'wires': [0]},
                                          {'input_idx': [1], 'func': 'ry', 'wires': [1]}])

        self.q_layer = self.Ansatz()

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=True):
        bsz = x.shape[0]
        # data = 2 * torch.arcsin(torch.sqrt(x[:, 0] + x[:, 1] - 2 * x[:, 0] * x[:, 1])).reshape(bsz, 1)
        data = x.reshape(bsz, 2)

        if use_qiskit:
            data = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, data)
        else:
            self.encoder(self.q_device, data)
            self.q_layer(self.q_device)
            data = self.measure(self.q_device)

        data = data.reshape(bsz, 2)

        return data


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = Simple2Class()
dataflow = dict()
for split in dataset:
    sampler = torch.utils.data.RandomSampler(dataset[split])
    dataflow[split] = torch.utils.data.DataLoader(
        dataset[split],
        batch_size=6,
        sampler=sampler,
        num_workers=8,
        pin_memory=True)


def build_noise_model_from_properties_and_config(properties,basis_gates):
    my_noise_model = NoiseModel(basis_gates=basis_gates)
    

    for qubits, error in basic_device_readout_errors(properties):
        my_noise_model.add_readout_error(error, qubits)

    gate_errors = basic_device_gate_errors(properties)
    for name, qubits, error in gate_errors:
        my_noise_model.add_quantum_error(error, name, qubits)
    return my_noise_model

import math





def get_loss(dataflow, model,flag):
    param_list = []
    for param in model.parameters():
        param_list.append(param)
    # print(param_list)

    target_all = []
    output_all = []
    for feed_dict in dataflow['valid']:
        inputs = feed_dict['data'].to(device)
        targets = feed_dict['target'].to(device)
        outputs = model(inputs,flag)
        # prediction = F.log_softmax(outputs, dim=1)
        target_all.append(targets)
        output_all.append(outputs)
    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)
    # loss = F.cross_entropy(output_all,target_all)
    print(output_all.mean())
    print('output_all:',output_all.mean())
    return output_all.mean()
        



def noise_injection(model,flag):
    circ = tq2qiskit(model.q_device, model.q_layer)
    circ.measure_all()
    circ_transpiled = processor.transpile(circs=circ)
    # print(circ_transpiled)
    print('circuit length:',len(circ_transpiled))
    q_layer = qiskit2tq(circ=circ_transpiled)
    model.measure.set_v_c_reg_mapping(get_v_c_reg_mapping(circ_transpiled))
    model.q_layer = q_layer
    if flag:
        noise_model_tq = NoiseModelTQ(
            noise_model= test_noise_model,
            n_epochs=5,
            noise_total_prob=0.5,
            factor=0.1,
            add_thermal=True
        )

        
        noise_model_tq.is_add_noise = True
        noise_model_tq.v_c_reg_mapping = get_v_c_reg_mapping(
            circ_transpiled)
        noise_model_tq.p_c_reg_mapping = get_p_c_reg_mapping(
            circ_transpiled)
        noise_model_tq.p_v_reg_mapping = get_p_v_reg_mapping(
            circ_transpiled)
        
        model.set_noise_model_tq(noise_model_tq)
    return model




def get_losses_noise(dataflow,flag):
    losses = torch.zeros([11,11])
    pi = math.pi
    p1= np.arange(-pi,pi*6/5, pi/5)
    p2= np.arange(-pi,pi*6/5 ,pi/5)
    # p1 = np.arange(-0.1,0.12,0.02)
    # p2 = np.arange(-0.1,0.12 ,0.02)
    print(p1)
    print(p2)
    for i in range(11):
        for j in range(11):
            #update parameter
            model = Q2Model().to(device)
            param_list = []
            for param in model.parameters():
                param_list.append(param)
            para1 = torch.nn.Parameter(torch.tensor(p1[i]) )
            para2 = torch.nn.Parameter(torch.tensor(p2[j]))
            param_list[0].copy_(para1)
            param_list[1].copy_(para2)
            #get loss
            # model = noise_injection(model,flag)
            model.set_qiskit_processor(processor)
            loss = get_loss(dataflow, model,flag)
            losses[i,j]= loss
    return p1,p2,losses

import pandas as pd
# buildup hu noise model
# provider = IBMQ.load_account()
# factory =  IBMMachinePropertiesFactory(provider)
# backend_name = 'ibmq_belem'
# config,properties,backend =  factory.query_backend_info(backend_name)
# test_datatime =  datetime.datetime.fromisoformat('2022-05-29 01:34:52-04:00')

# test_property = backend.properties(datetime=test_datatime)

# test_noise_model =   build_noise_model_from_properties_and_config(test_property,config.basis_gates) 
# processor = QiskitProcessor(use_real_qc=False,noise_model= test_noise_model,noise_model_name=backend.name(),initial_layout=[0,1])


# with torch.no_grad():
#     X,Y,loss_perfect_all = get_losses_noise(dataflow,False)
# print(loss_perfect_all)
# torch.save(loss_perfect_all,'loss_perfect_{}.npy'.format(4))


# with torch.no_grad():
#     X,Y,loss_noise_all = get_losses_noise(dataflow,True)
# print(loss_noise_all)
# torch.save(loss_noise_all,'loss_noise_{}.npy'.format(4))


# data = dict()
# data['x'] =X
# data['loss_perfect'] = loss_perfect
# data['loss_noise'] = loss_noise
# df = pd.DataFrame.from_dict(data)
# df.to_excel('aaa.xlsx')
# print


pi =math.pi
X = np.arange(-0.1,0.12,0.02)
Y = np.arange(-0.1,0.12 ,0.02)
# X= np.arange(-pi,pi*6/5, pi/5)
# Y= np.arange(-pi,pi*6/5 ,pi/5)


loss_perfect_all = torch.load('loss_perfect_3.npy')
loss_noise_all = torch.load('loss_noise_3.npy')
loss_diff_all = torch.abs(loss_perfect_all- loss_noise_all)

xx, yy = np.meshgrid(X, Y)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_perfect_all.reshape(newshape)


sns.set(style='white')


fig = plt.figure()
# plt.rcParams["font.family"] = "Times New Roman"

ax = fig.add_subplot(131, projection='3d')
norm = matplotlib.colors.Normalize(0.75,0.9)


ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)

aa1 = list(range(5))
aa2 = list(range(6,11))

ax = fig.add_subplot(132, projection='3d')
norm = matplotlib.colors.Normalize(0.6,0.8)


Z1 = loss_noise_all[:,5]
Z2 = loss_noise_all[5,:]

X2 = X[aa1]
Y2 =Y[aa1]
loss_noise_surface = loss_noise_all[aa1][:,aa1]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)

X2 = X[aa1]
Y2 =Y[aa2]
loss_noise_surface = loss_noise_all[aa1][:,aa2]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)

X2 = X[aa2]
Y2 =Y[aa1]
loss_noise_surface = loss_noise_all[aa2][:,aa1]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)

X2 = X[aa2]
Y2 =Y[aa2]
loss_noise_surface = loss_noise_all[aa2][:,aa2]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)


ax.scatter(X,Y[5],Z1,cmap=cm.coolwarm,norm = norm)
ax.scatter(X[5],Y,Z2,cmap=cm.coolwarm,norm = norm)



ax = fig.add_subplot(133, projection='3d')
norm = matplotlib.colors.Normalize(0,0.2)


Z1 = loss_diff_all[:,5]
Z2 = loss_diff_all[5,:]

X2 = X[aa1]
Y2 =Y[aa1]
loss_noise_surface = loss_diff_all[aa1][:,aa1]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)



X2 = X[aa1]
Y2 =Y[aa2]
loss_noise_surface = loss_diff_all[aa1][:,aa2]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)

X2 = X[aa2]
Y2 =Y[aa1]
loss_noise_surface = loss_diff_all[aa2][:,aa1]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)

X2 = X[aa2]
Y2 =Y[aa2]
loss_noise_surface = loss_diff_all[aa2][:,aa2]
xx, yy = np.meshgrid(X2, Y2)
newshape = (xx.shape[0])*(yy.shape[0])
x_input = xx.reshape(newshape)
y_input = yy.reshape(newshape)
z_input = loss_noise_surface.reshape(newshape)
ax.plot_trisurf(x_input,y_input,z_input,cmap=cm.coolwarm,norm = norm)


ax.scatter(X,Y[5],Z1,cmap=cm.coolwarm,norm = norm)
ax.scatter(X[5],Y,Z2,cmap=cm.coolwarm,norm = norm)


plt.savefig('aaaaaa.eps')
plt.show()