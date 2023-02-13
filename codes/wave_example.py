from distutils.command.build_scripts import first_line_re
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq
import torchquantum.functional as tqf
import random
from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset,DataLoader
import sys
from q_layers import QLayer22, QLayer4,QLayer1

import numpy as np
# from wave_qmodel import QFCModel,QRNN,QRNNBlockVqc,MultiOutputQRNNBlockVqc2
import copy
data_path = 'waveforms2500.pk'
data_path2 = 'waveforms1500'

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data = np.load(data_path,allow_pickle=True)
x =data[0]
max_data = np.max(np.max(np.abs(x)*1e6,axis=1),axis=1)
print(max_data.shape)
print(np.logical_and(max_data>5,max_data < 10 ).sum())
# print((max_data<5).sum() )
# print((max_data>10 ).sum() )
y = data[1]

print(x.shape)
print(y.shape)


print((y==1).sum())
print((y==0).sum())
min_num =2
max_num =13
print(np.logical_and(max_data<min_num,y==0 ).sum())
print(np.logical_and(max_data>max_num,y==1 ).sum())
real_false =np.logical_and(max_data<min_num,y==0 )
real_true =np.logical_and(max_data>max_num,y==1 )
real_data = np.logical_or(real_false,real_true)
x = x[real_data][:1500]
y = y[real_data][:1500]
print(x.shape)
print(y.shape)

data2 = dict()

data2['data'] =x
data2['target'] =y
np.save(data_path2,data2)

sys.exit(0)
data_num =1500
train_rate = 0.9
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

class WaveIntegrateDataset(Dataset):
    def __init__(self,path,times=1,num_block=1,index = None,window_size =None):
        self.data = np.load(path,allow_pickle=True)
        self.X = self.data[0][real_data]
        self.Y = self.data[1][real_data]
        if index is not None:
            self.X =self.X[index]
            self.Y = self.Y[index]
        self.channel = self.X.shape[-1]
        self.X = self.X * 1e5
        self.X = abs(self.X)
        self.X = np.swapaxes(self.X,1,2)
        self.X = self.X[:,1,:]
        self.len = num_block *window_size
        self.samples = len(self.Y)
        print(self.X.shape)
        self.X = self.X[:,100:1000]
        self.X = self.X[:,:self.len]
        print(self.X.shape)
        self.X = self.X.reshape(self.samples,num_block,window_size)
        self.integrate = np.sum(self.X,axis=-1)
        self.times = times

    def __len__(self):
        return self.times * self.samples

    def __getitem__(self, idx):
        x = self.integrate[idx%self.samples]
        x = torch.tensor(x,dtype=torch.float32)
        y = self.Y[idx%self.samples]
        return x,y




def train(dataflow, model, device, optimizer):
    target_all = []
    output_all = []
    
    for batch_idx, (data, target) in enumerate(dataflow):
        inputs = data.to(device)
        targets = target.to(device)

        outputs = model(inputs)

        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

def valid_test(dataflow, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataflow):
            inputs = data.to(device)
            targets = target.to(device)
            outputs = model(inputs)

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
    return accuracy



class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self,wire):
            super().__init__()
            self.n_wires = wire
            self.layer1 = QLayer4(wire)
            self.layer2 = QLayer4(wire)
            # self.layer3 = QLayer4(wire)
            # self.layer4 = QLayer4(wire)           
                        
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            self.layer1(self.q_device)
            self.layer2(self.q_device)
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
        x = x.reshape(bsz, 16)
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    args = parser.parse_args()
    random_index = list(range(0,data_num))
    random.shuffle(random_index)
    train_index = random_index[:int(data_num*train_rate)]
    test_index = random_index[int(data_num*train_rate):]

    train_db = WaveIntegrateDataset(data_path,times=1,num_block=16,index=train_index,window_size =50)
    test_db = WaveIntegrateDataset(data_path,times=1,num_block=16,index=test_index,window_size =50)

    train_data = DataLoader(train_db, batch_size=64, shuffle=True)
    test_data = DataLoader(test_db, batch_size=64, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    model = QFCModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr = 0.005)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_acc =0

    if args.static:
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    for epoch in range(1, n_epochs + 1):
        # train
        train(train_data, model, device, optimizer)
        acc = valid_test(test_data, model, device)

        if best_acc<acc:
            best_acc = acc
        print(f'Epoch {epoch}: current acc = {acc},best acc = {best_acc}')
        scheduler.step()

if __name__ == '__main__':
    sys.stdout = Logger('./logger/{}_log.txt'.format('block3_xyz_window_size_test'))
    main()
