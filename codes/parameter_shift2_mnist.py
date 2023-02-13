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


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self,wire):
            super().__init__()
            self.n_wires = wire
            self.layer1 = VQC3_block()
            # self.layer2 = VQC3_block()


        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            self.layer1(self.q_device)
            # self.layer2(self.q_device)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
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

def shift_and_run(model, inputs, use_qiskit=False):
    param_list = []
    for param in model.parameters():
        param_list.append(param)
    grad_list = []
    for param in param_list:
        param.copy_(param + np.pi * 0.5)
        out1 = model(inputs, use_qiskit)
        param.copy_(param - np.pi)
        out2 = model(inputs, use_qiskit)
        param.copy_(param + np.pi * 0.5)
        grad = 0.5 * (out1 - out2)
        grad_list.append(grad)
    return model(inputs, use_qiskit), grad_list

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# dataset = Simple2Class()
# dataflow = dict()
# for split in dataset:
#     sampler = torch.utils.data.RandomSampler(dataset[split])
#     dataflow[split] = torch.utils.data.DataLoader(
#         dataset[split],
#         batch_size=6,
#         sampler=sampler,
#         num_workers=8,
#         pin_memory=True)

def train_2qubit(dataflow, model, device, optimizer, qiskit=False, input_name = 'image', target_name = 'digit'):
    train_times =0
    for feed_dict in dataflow['train']:
        inputs = feed_dict[input_name].to(device)
        targets = feed_dict[target_name].to(device)

        print('train_times:',train_times,'\n')
        train_times = train_times+1
        start_time1 = time.time()
        with torch.no_grad():
            outputs, grad_list = shift_and_run(model, inputs, use_qiskit=qiskit)
        end_time1 = time.time()
        one_batch_time = end_time1 -start_time1
        print('one batch time = {}'.format(one_batch_time))

        outputs.requires_grad=True
        prediction = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        for i, param in enumerate(model.parameters()):
            param.grad = torch.sum(grad_list[i] * outputs.grad).to(dtype=torch.float32, device=param.device).view(param.shape)
        optimizer.step()
        # print(f"loss: {loss.item()}", end='\r')


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
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")


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

dataset = MNIST(
    root='./mnist_data',
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest=[3,6],
    n_test_samples=320,
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

model = QFCModel().to(device)


backend = FakeJakarta()
processor_real_qc = QiskitProcessor(use_real_qc=False,backend = backend,noise_model= None)
model.set_qiskit_processor(processor_real_qc)

n_epochs = 1
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

total_time = 0

for epoch in range(1, n_epochs + 1):
    # train
    print(f"Epoch {epoch}:")
    start_time = time.time()
    train_2qubit(dataflow, model, device, optimizer, qiskit=True)
    end_time  = time.time()
    training_time = end_time - start_time
    total_time = total_time + training_time
    print(optimizer.param_groups[0]['lr'])
    # valid
    valid_test_2qubit(dataflow, 'test', model, device, qiskit=True)
    scheduler.step()
    print("Time cost {:.2f}s".format(training_time))
print("Total Time cost {:.2f}s".format(total_time))

# backend = FakeQuito()
processor_real_qc = QiskitProcessor(use_real_qc=False,backend = backend,noise_model= None)
model.set_qiskit_processor(processor_real_qc)
# test
valid_test_2qubit(dataflow, 'test', model, device, qiskit=True)



