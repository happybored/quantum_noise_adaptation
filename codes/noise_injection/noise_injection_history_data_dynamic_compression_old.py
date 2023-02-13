import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

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
from my_property import IBMMachinePropertiesFactory,MyBackendNoise,CalibDictionary
import warnings
import math
from qcompression import ADMM,get_model_depth
import copy
from qiskit.providers.aer.noise.device.models import (basic_device_readout_errors,
                                                      basic_device_gate_errors)

from qiskit.providers.aer.noise import NoiseModel
warnings.filterwarnings("ignore")



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


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

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
        input = feed_dict['image'].to(device)
        target = feed_dict['digit'].to(device)
       
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
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

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
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

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


df3 = pd.read_csv('calibration_data/prune_ratios.xlsx' )
df3 = df3[['timestamp','pr']]
prune_ratio = dict()
for i in range(len(df3)):
    key = df3['timestamp'].loc[i].split(' ')[0]
    prune_ratio[key] = df3['pr'].loc[i]



def main(train_noise_model,test_noise_model,backend,train_situation,time=None):


    print(prune_ratio)
    print(train_situation['query_model'])
    if train_situation['query_model'] in prune_ratio.keys():
        print('find model in prune dict!')
        pr =  prune_ratio[train_situation['query_model']]
    else:
        print('no model in prune dict!')
        pr = 0.4

    

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = MNIST(
        root='./mnist_data',
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[0,1,3,6],
        n_test_samples=200,
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


    
    processor = QiskitProcessor(use_real_qc=False,backend=backend,noise_model_name=backend.name()) 

    circ_transpiled = processor.transpile(circs=circ)
    q_layer = qiskit2tq(circ=circ_transpiled)

    model.measure.set_v_c_reg_mapping(
        get_v_c_reg_mapping(circ_transpiled))
    model.q_layer = q_layer

    noise_model_tq = NoiseModelTQ(
        noise_model= train_noise_model,
        n_epochs=n_epochs,
        noise_total_prob=0.5,
        # ignored_ops=configs.trainer.ignored_noise_ops,
        factor=0.1,
        add_thermal=True
    )

    noise_model_tq.is_add_noise = True
    # noise_model_tq.v_c_reg_mapping = {'v2c': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    #                                   'c2v': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    #                                   }
    # noise_model_tq.p_c_reg_mapping = {'p2c': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    #                                   'c2p': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    #                                   }
    # noise_model_tq.p_v_reg_mapping ={'p2v': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    #                                   'v2p': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
    #                                   }
    noise_model_tq.v_c_reg_mapping = get_v_c_reg_mapping(
        circ_transpiled)
    noise_model_tq.p_c_reg_mapping = get_p_c_reg_mapping(
        circ_transpiled)
    noise_model_tq.p_v_reg_mapping = get_p_v_reg_mapping(
        circ_transpiled)

    model.set_noise_model_tq(noise_model_tq)

    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    train_flag = train_situation["train_flag"]
    if train_flag:

        for epoch in range(1, n_epochs + 1):
            # train
            print(f"Epoch {epoch}:")
            train(dataflow, model, device, optimizer)
            # print(optimizer.param_groups[0]['lr'])
    
            # valid
            valid(dataflow, 'valid', model, device)
            scheduler.step()
        # torch.save(model,'models/model_{}.pth'.format(time))
        # torch.save(model,'models/model.pth')

        # model = torch.load('models/model.pth')


        original_depth = get_model_depth(model,backend)
        original_acc = test(model,dataflow)
        print('The transpiled circuit length of original model is {}\n. Accuracy is {}'.format(original_depth,original_acc) )



        #ADMM Training
        class ADMMArgs:
          def __init__(self):
            self.admm = True
            self.masked_retrain = True
            self.rho =0.001
            self.admm_epochs = 1
            self.rho_num =1
            self.lr = 0.01
            self.epochs = 5
            self.prune_ratio =  pr
    
        admm_args = ADMMArgs()
        admm_criterion = torch.nn.CrossEntropyLoss().to(device)
        
        if admm_args.admm:
            print('*'*20 +'admm'+'*'*20)
            print('prune rate = ',admm_args.prune_ratio)
            optimizer = torch.optim.Adam(model.parameters(), admm_args.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
            initial_rho = admm_args.rho
            for i in range(admm_args.rho_num):
                current_rho = initial_rho * 10 ** i
                # print("current rho: {}".format(current_rho))
                admm = ADMM(model, admm_args, rho=current_rho)
                admm.admm_initialization(model=model)  # intialize Z variabl
                # admm train
                best_prec1 = 0.
                for epoch in range(1, admm_args.epochs + 1):
                    train_admm(model, dataflow, admm_criterion, optimizer, scheduler, epoch,  mask=None,admm_flag= True,admm= admm)
                    prec1 = test(model,dataflow)
                    best_prec1 = max(prec1, best_prec1)
                    print("Best Acc: {:.4f}%".format(best_prec1))

        if admm_args.masked_retrain:
            print('*'*20 +'masked_retrain'+'*'*20)
            optimizer = torch.optim.Adam(model.parameters(), admm_args.lr)
            scheduler = None
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
            model.to(device)
            masks = admm.hard_prune(model)
            epoch_loss_dict = {}
            testAcc = []
            testDepth = []
            best_prec1 = 0
            best_depth1 = 0
            best_metrics1 =[0]
            for epoch in range(1, admm_args.epochs + 1):
                idx_loss_dict = train_admm(model, dataflow, admm_criterion, optimizer, scheduler, epoch,mask = masks,admm_flag=False)
                prec1 = test(model,dataflow)
                temp_model  = copy.deepcopy(model)
                depth1 = get_model_depth(temp_model,backend) 
                prec_norm = prec1*1.0/original_acc
                depth_norm =depth1*1.0/original_depth
                para_metrics1 = prec_norm + 1.0/depth_norm
                if para_metrics1 > max(best_metrics1):
                    best_model = copy.deepcopy(model)
                    best_prec1 = prec1
                    best_depth1 = depth1
                    print(">_ Got better accuracy {:.3f}% now...".format(prec1))
                epoch_loss_dict[epoch] = idx_loss_dict
                testAcc.append(prec1)
                testDepth.append(depth1)
                best_metrics1.append(para_metrics1)
            print("Best Acc: {:.4f}% , Best Depth: {:d}".format(best_prec1,best_depth1))
        torch.save(best_model,'dynamic_dict/{}'.format(train_situation['query_model']))
    else:
        model = torch.load('dynamic_dict/{}'.format(train_situation['query_model']))


    # test
    valid(dataflow, 'test', model, device, qiskit=False)


    print(f"\nTest with Qiskit Simulator")
    processor_simulation = QiskitProcessor(use_real_qc=False,noise_model= test_noise_model,noise_model_name=backend.name())
    model.set_qiskit_processor(processor_simulation)
    acc,confidence = valid(dataflow, 'test', model, device, qiskit=True)
    return acc,confidence




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
args = parser.parse_args()



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
    print(path)
    # df = pd.read_csv(path)
    df = pd.read_excel(path)
    df = pd.DataFrame(df)
    df2 = df.copy(deep=True)
    calib_dict = CalibDictionary()
    df2['dynamic2'] = 0
    # df2['confidence'] = 0
    df2['query_model'] = 0
    df2['train_flag'] = 0

    date_format =  "%Y-%m-%d-%H-%M"
    train_datatime =  datetime.datetime.fromisoformat(df['timestamp'].loc[0])
    train_property = backend.properties(datetime=train_datatime)
    my_backend_noise=  MyBackendNoise(backend_name,timestamp=train_datatime)
    my_backend_noise.set_property(train_property)
    train_noise = my_backend_noise.get_noise()
    train_noise_model = build_noise_model_from_properties_and_config(train_property,config.basis_gates)

    print('-'*20+df['timestamp'].loc[0]+'-'*20)
    
    #get calibration
    
    for i in range(0,len(df2)):
        
        test_datatime =  datetime.datetime.fromisoformat(df2['timestamp'].loc[i])
        print()
        print('-'*20+df2['timestamp'].loc[i]+'-'*20)
        test_property = backend.properties(datetime=test_datatime)
        my_backend_noise=  MyBackendNoise(backend_name,timestamp=test_datatime)
        my_backend_noise.set_property(test_property)
        test_noise = my_backend_noise.get_noise()
        test_property_model =   build_noise_model_from_properties_and_config(test_property,config.basis_gates)
        # test_model_path = ''  # designed by cluster 
        train_situation = calib_dict.process(test_noise)


        acc,confidence = main(train_noise_model,test_property_model,backend,train_situation,df2['timestamp'].loc[i])
        df2['dynamic2'].loc[i] = acc
        # df2['confidence'].loc[i] = confidence
        df2['query_model'].loc[i]= train_situation['query_model']
        df2['train_flag'].loc[i] = train_situation['train_flag']

        df2.to_excel(path)
    print(calib_dict.model_dict.keys()) 
    df2.to_excel(path)
    ##save result