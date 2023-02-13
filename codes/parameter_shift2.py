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

# change qiskit processor to fake/ zidingyi backend

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
            print(self.target[-num:])

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
            self.n_wires = 2
            self.op1 = tq.RZ(has_params=True, trainable=True)
            self.op2 = tq.RY(has_params=True, trainable=True)
            self.op3 = tq.RY(has_params=True, trainable=True)
            self.op4 = tq.CNOT(has_params=False, trainable=False)
        
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.op1(self.q_device, wires=0)
            self.op2(self.q_device, wires=1)
            self.op3(self.q_device, wires=0)
            self.op4(self.q_device, wires=[0, 1])

    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([{'input_idx': [0], 'func': 'ry', 'wires': [0]}])

        self.ansatz = self.Ansatz()

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        data = 2 * torch.arcsin(torch.sqrt(x[:, 0] + x[:, 1] - 2 * x[:, 0] * x[:, 1])).reshape(bsz, 1)

        if use_qiskit:
            data = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.ansatz, self.measure, data)
        else:
            self.encoder(self.q_device, data)
            self.ansatz(self.q_device)
            data = self.measure(self.q_device)

        data = data.reshape(bsz, 2)

        return data

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

def train_2qubit(dataflow, model, device, optimizer, qiskit=False, input_name = 'data', target_name = 'target'):
    train_times =0
    for feed_dict in dataflow['train']:
        inputs = feed_dict[input_name].to(device)
        targets = feed_dict[target_name].to(device)

        print('train_times:',train_times,'\n')
        train_times = train_times+1

        with torch.no_grad():
            outputs, grad_list = shift_and_run(model, inputs, use_qiskit=qiskit)
        outputs.requires_grad=True
        prediction = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        for i, param in enumerate(model.parameters()):
            param.grad = torch.sum(grad_list[i] * outputs.grad).to(dtype=torch.float32, device=param.device).view(param.shape)
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')


def valid_test_2qubit(dataflow, split, model, device, qiskit=False, input_name = 'data', target_name = 'target'):
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
model = Q2Model().to(device)
backend = FakeJakarta()
processor_real_qc = QiskitProcessor(use_real_qc=False,backend = backend,noise_model= None)
model.set_qiskit_processor(processor_real_qc)

n_epochs = 5
optimizer = optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
for epoch in range(1, n_epochs + 1):
    # train
    print(f"Epoch {epoch}:")
    train_2qubit(dataflow, model, device, optimizer, qiskit=True)
    print(optimizer.param_groups[0]['lr'])
    # valid
    valid_test_2qubit(dataflow, 'valid', model, device, qiskit=True)
    scheduler.step()

backend = FakeQuito()
processor_real_qc = QiskitProcessor(use_real_qc=False,backend = backend,noise_model= None)
model.set_qiskit_processor(processor_real_qc)
# test
valid_test_2qubit(dataflow, 'test', model, device, qiskit=True)



