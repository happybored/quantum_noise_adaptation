from torchpack.datasets.dataset import Dataset
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.layers import SethLayer0

from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from q_layers import QLayer22, QLayer4,QLayer1,VQC3_block

# change qiskit processor to fake/ zidingyi backend


from modified.qiskit_processor import QiskitProcessor
from qiskit.test.mock import FakeValencia,FakeQuito,FakeJakarta
import random
import time
import datetime    
import sys

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self,wire):
            super().__init__()
            self.n_wires = wire
            self.layer1 = VQC3_block(wire)
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.layer1(self.q_device)

    def __init__(self,wires=4):
        super().__init__()
        self.n_wires = wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        # print(x.shape)
        x = x[:,0:4].reshape(bsz, 4, 1).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

def valid_test_2qubit(dataflow, split, model, device, qiskit=False, input_name = 'image', target_name = 'digit'):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict[input_name].to(device)
            targets = feed_dict[target_name].to(device)

            outputs = model(inputs, use_qiskit=qiskit)
            prediction = F.log_softmax(outputs, dim=1)

            target_all.append(targets)
            output_all.append(prediction)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




dataset = MNIST(
    root='./mnist_data',
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest=[3,6],
    n_test_samples=64,
)

dataflow = dict()
for split in dataset:
    sampler = torch.utils.data.RandomSampler(dataset[split])
    dataflow[split] = torch.utils.data.DataLoader(
        dataset[split],
        batch_size=64,
        sampler=sampler,
        num_workers=8,
        pin_memory=True)
    print('{}\'s data length:{}'.format(split,len(dataset[split])))
# sys.exit(0)



def main(wires):
    model = QFCModel(wires).to(device)
    n_epochs = 5
    # if wires<=5:
    #     backend_name = 'ibmq_belem'
    if wires<= 7:
        backend_name = 'ibm_lagos'
    elif wires<= 16:
        backend_name = 'ibmq_guadalupe'
    elif wires<= 27:
        backend_name = 'ibmq_montreal'
    elif wires> 27:
        backend_name = 'ibm_washington'

    processor_real_qc = QiskitProcessor(use_real_qc=True,backend_name=backend_name,hub='ibm-q-lanl', group='lanl', project='quantum-optimiza',initial_layout= list(range(wires)) )
    # processor_simulation = QiskitProcessor(use_real_qc=False,noise_model_name= backend_name,hub='ibm-q-lanl', group='lanl', project='quantum-optimiza',initial_layout= list(range(wires)))
    model.set_qiskit_processor(processor_real_qc)
    total_time = 0
    
    print('start'+50*'-')
    for i in range(n_epochs):
        start_time = time.time()
        valid_test_2qubit(dataflow, 'test', model, device, qiskit=True)
        end_time  = time.time()
        training_time = end_time - start_time
        total_time = total_time + training_time
        print("Total Time cost {:.2f}s".format(total_time))
    return total_time

if __name__ == '__main__':
    import pandas as pd
    import os
    fname = 'time_records/time_records_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
    wires = [9,10,11,12,13,14,15,16]
    total_times = []
    for wire in wires:
        print('wire = ',wire)
        total_time = main(wire)
        total_times.append(total_time)
        result = {'wire':[wire],'time':[total_time]}
        df = pd.DataFrame.from_dict(result)

        
        if os.path.isfile(fname):
            df.to_csv(fname, mode = 'a', index = False, header = False)
        else:
            df.to_csv(fname, mode = 'w', index = False, header = True)        

    totals =np.array([wires,total_times])
    np.savetxt('time_records/time_records_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")),totals)
    
    print(total_times)
