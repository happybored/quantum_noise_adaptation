import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchpack.datasets.dataset import Dataset

import torchquantum as tq
import torchquantum.functional as tqf
import sys
from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchquantum.plugins import tq2qiskit, qiskit2tq
from torchquantum.utils import (build_module_from_op_list,
                                build_module_op_list,
                                get_v_c_reg_mapping,
                                get_p_c_reg_mapping,
                                get_p_v_reg_mapping,
                                get_cared_configs)

from modified.qiskit_processor import QiskitProcessor
# from qiskit.test.mock import FakeQuito,FakeBelem
from modified.noise_model import NoiseModelTQ
from q_layers import QLayer22, QLayer4,QLayer1
import random
import numpy as np
import pandas as pd
from my_property import IBMMachinePropertiesFactory,MyBackendNoise
import warnings
import math
from qcompression import ADMM,get_model_depth
import copy
from qiskit.providers.aer.noise.device.models import (basic_device_readout_errors,
                                                      basic_device_gate_errors)
import os
from qiskit.providers.aer.noise import NoiseModel
warnings.filterwarnings("ignore")

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class Logger(object):
    def __init__(self, file_path: str = "./Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")
        self.encoding = None

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self,wire):
            super().__init__()
            self.n_wires = wire
            self.layer1 = QLayer4()
            self.layer2 = QLayer4()
            self.layer3 = QLayer4()
            # self.layer4 = QLayer4()
            # self.layer5 = QLayer4(3)
            # self.layer6 = QLayer4(3)
            # self.layer7 = QLayer4(3)
            # self.layer8 = QLayer4(3)
            # self.layer9 = QLayer4(3)
                        
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            self.layer1(self.q_device)
            self.layer2(self.q_device)
            self.layer3(self.q_device)
            # self.layer4(self.q_device)
            # self.layer5(self.q_device)
            # self.layer6(self.q_device)
            # self.layer7(self.q_device)
            # self.layer8(self.q_device)
            # self.layer9(self.q_device)


    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4_ry'])

        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        # x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x[:,0:3].reshape(bsz, 3, 1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x



def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['data'].to(device)
        targets = feed_dict['target'].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')


def train_admm(model,dataflow,criterion, optimizer, scheduler, epoch, mask =None,admm_flag = True, admm = None,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # switch to train mode
    model.to(device)
    model.train()

    masks = mask
    for i, feed_dict in enumerate(dataflow['train']):
        input = feed_dict['data'].to(device)
        target = feed_dict['target'].to(device)
       
        # adjust learning rate
        if admm_flag:
            admm.admm_adjust_learning_rate(optimizer, epoch)
        else:
            scheduler.step()

        # compute output
        output = model(input)
        ce_loss = criterion(output, target)

        if admm_flag:
            admm.z_u_update(model, epoch, i)  # update Z and U variables
            ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(model, ce_loss)  # append admm losss

        optimizer.zero_grad()

        if admm_flag:
            mixed_loss.backward()
        else:
            ce_loss.backward()

        if masks != None:
            mask_index =0
            with torch.no_grad():
                for item in model.parameters():
                    device_mask = masks[mask_index].to(device)
                    item.grad *= device_mask
                    mask_index = mask_index+1

        optimizer.step()

def test(model, dataflow, split = 'valid',  device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['data'].to(device)
            targets = feed_dict['target'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    return accuracy



def valid(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['data'].to(device)
            targets = feed_dict['target'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    confidence = np.exp(-loss)
    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set confidence: {confidence}")
    print(f"{split} set loss: {loss}")
    return accuracy,confidence

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.34, random_state=2)

print(len(y_train))
print(len(y_test))

class IrisTrainDataset(torch.utils.data.Dataset):
    def __init__(self, num=11):
        self.data = torch.from_numpy(X_train).float()
        self.target = torch.from_numpy(y_train).long()
       
    def __getitem__(self, idx):
        return {'data': self.data[idx], 'target': self.target[idx]}

    def __len__(self):
        return len(self.target) - 1

class IrisTestDataset(torch.utils.data.Dataset):
    def __init__(self, num=11):
        self.data = torch.from_numpy(X_test).float()
        self.target = torch.from_numpy(y_test).long()
       
    def __getitem__(self, idx):
        return {'data': self.data[idx], 'target': self.target[idx]}

    def __len__(self):
        return len(self.target) - 1

class Simple2Class(Dataset):
    def __init__(self):
        train_dataset = IrisTrainDataset()
        valid_dataset = IrisTestDataset(num=10)
        datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': valid_dataset}
        super().__init__(datasets)
        

def main(train_noise_model,test_noise_model,backend,time,qubit_noise =None,pruning_rate =None):



    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    
    dataset = Simple2Class()
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=64,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QFCModel().to(device)
    # noise_model_tq = builder.make_noise_model_tq()

    n_epochs = args.epochs

    # from qiskit import IBMQ
    # IBMQ.load_account()

    circ = tq2qiskit(model.q_device, model.q_layer)
    """
    add measure because the transpile process may permute the wires, 
    so we need to get the final q reg to c reg mapping 
    """
    circ.measure_all()


    
    processor = QiskitProcessor(use_real_qc=False,backend=backend,noise_model_name=backend.name(),initial_layout=[0,1,2,3]) 

    circ_transpiled = processor.transpile(circs=circ)
    q_layer = qiskit2tq(circ=circ_transpiled)

    model.measure.set_v_c_reg_mapping(
        get_v_c_reg_mapping(circ_transpiled))
    model.q_layer = q_layer



#set noise model
    # noise_model_tq = NoiseModelTQ(
    #     noise_model= train_noise_model,
    #     n_epochs=n_epochs,
    #     noise_total_prob=0.5,
    #     # ignored_ops=configs.trainer.ignored_noise_ops,
    #     factor=0.1,
    #     add_thermal=True
    # )
    # noise_model_tq.is_add_noise = True
    # noise_model_tq.v_c_reg_mapping = get_v_c_reg_mapping(
    #     circ_transpiled)
    # noise_model_tq.p_c_reg_mapping = get_p_c_reg_mapping(
    #     circ_transpiled)
    # noise_model_tq.p_v_reg_mapping = get_p_v_reg_mapping(
    #     circ_transpiled)
    # model.set_noise_model_tq(noise_model_tq)



    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    global train_flag
    if train_flag:
        # train_flag = False
        for epoch in range(1, n_epochs + 1):
            # train
            print(f"Epoch {epoch}:")
            train(dataflow, model, device, optimizer)
            # print(optimizer.param_groups[0]['lr'])
    
            # valid
            valid(dataflow, 'valid', model, device)
            scheduler.step()

        torch.save(model,'database_iris/{}/{}.pth'.format(args.pruning_type,time))
    else:
        pass
        # model = torch.load('dynamic_model/{}'.format(args.modelpath))



    # temp data
    # circ = tq2qiskit(model.q_device, model.q_layer)
    # circ.measure_all()
    # processor = QiskitProcessor(use_real_qc=False,backend=backend,noise_model_name=backend.name()) 
    # circ_transpiled = processor.transpile(circs=circ)
    # circ_transpiled.draw(output='mpl',filename='circ.png')

    # test
    valid(dataflow, 'test', model, device, qiskit=False)


    print(f"\nTest with Qiskit Simulator")
    processor_simulation = QiskitProcessor(use_real_qc=False,noise_model= test_noise_model,noise_model_name=backend.name(),initial_layout=[0,1,2,3])
    model.set_qiskit_processor(processor_simulation)
    acc,confidence = valid(dataflow, 'test', model, device, qiskit=True)
    result =  dict()
    result['timestamp'] =[time]
    result['accuracy'] = [acc]
    result['confidence'] = [confidence]
    return result




def build_noise_model_from_properties_and_config(properties,basis_gates):
    my_noise_model = NoiseModel(basis_gates=basis_gates)
    

    for qubits, error in basic_device_readout_errors(properties):
        my_noise_model.add_readout_error(error, qubits)

    gate_errors = basic_device_gate_errors(properties)
    for name, qubits, error in gate_errors:
        my_noise_model.add_quantum_error(error, name, qubits)
    return my_noise_model



parser = argparse.ArgumentParser()
parser.add_argument('--static', action='store_true', help='compute with '
                                                          'static mode')
parser.add_argument('--pdb', action='store_true', help='debug with pdb')
parser.add_argument('--wires-per-block', type=int, default=3,
                    help='wires per block int static mode')
parser.add_argument('--epochs', type=int, default=4,
                    help='number of training epochs')
parser.add_argument('--pr', type=float, default=0.5,
                    help='prune rate')
parser.add_argument('--excelpath', type=str, default='',
                    help='prune rate')
parser.add_argument('--modelpath', type=str, default='',
                    help='prune rate')
parser.add_argument('--acc_name', type=str, default='',
                    help='prune rate')
parser.add_argument('--pruning_type', type=str, default='noise_aware_pruning',
                    help='prune rate')
args = parser.parse_args()


train_flag = True

if __name__ == '__main__':

    sys.stdout = Logger('./logger/{}_log.txt'.format('test'))
    import datetime    
    ##load csv 
    from qiskit import IBMQ
    provider = IBMQ.load_account()
    factory =  IBMMachinePropertiesFactory(provider)
    backend_name = 'ibmq_belem'
    config,properties,backend =  factory.query_backend_info(backend_name)

    path = 'calibration_data/' + args.excelpath 
    # df = pd.read_csv(path)
    df = pd.read_excel(path)
    df = pd.DataFrame(df)

    df2 = df.copy(deep=True)
    df2[args.acc_name] = 0
    date_format =  "%Y-%m-%d-%H-%M"

    for i in range(0,len(df2)):
        test_datatime =  datetime.datetime.fromisoformat(df2['timestamp'].loc[i])
        qubit_noise = np.array(df2.loc[i][['id0','id1','id2','id3']])
        print()
        print('-'*20+df2['timestamp'].loc[i]+'-'*20)
        test_property = backend.properties(datetime=test_datatime)
        test_property_model =   build_noise_model_from_properties_and_config(test_property,config.basis_gates)     
        result = main(test_property_model,test_property_model,backend,df2['timestamp'].loc[i],qubit_noise = qubit_noise)
        df2[args.acc_name].loc[i] = result['accuracy'][0]
        df2.to_excel(path)      

    # df2.to_excel(path)
    ##save result