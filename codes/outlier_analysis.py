import pandas as pd
import numpy as np
import copy

path_train_alone = 'calibration_data/ibmq_belem-tech.csv'
output_path = 'analysis_pic/'
path_result = 'calibration_data/ibmq_belem-tech_constrain_test.xlsx'

df = pd.read_csv(path_train_alone)
keys = ['id0', 'id1', 'id2', 'id3', 'id4', 'cx4_3', 'cx3_4', 'cx3_1', 'cx1_3', 'cx2_1', 'cx1_2', 'cx1_0',
       'cx0_1', 'T1_q0', 'T2_q0', 'frequency_q0', 'readout_error_q0',
       'prob_meas0_prep1_q0', 'prob_meas1_prep0_q0', 'T1_q1', 'T2_q1',
       'frequency_q1', 'readout_error_q1', 'prob_meas0_prep1_q1',
       'prob_meas1_prep0_q1', 'T1_q2', 'T2_q2', 'frequency_q2',
       'readout_error_q2', 'prob_meas0_prep1_q2', 'prob_meas1_prep0_q2',
       'T1_q3', 'T2_q3', 'frequency_q3', 'readout_error_q3',
       'prob_meas0_prep1_q3', 'prob_meas1_prep0_q3', 'T1_q4', 'T2_q4',
       'frequency_q4', 'readout_error_q4', 'prob_meas0_prep1_q4',
       'prob_meas1_prep0_q4']
keys  = ['gate0','gate1','gate2','gate3', 'gate4','cx1_3', 'cx1_2', 'cx0_1','cx3_4','readout0','readout1','readout2','readout3']
keys2  = ['gate0','gate1','gate2','gate3', 'gate4','cx1_3', 'cx1_2','cx0_1','cx3_4']

thres = {}
for i in range(5):
    thres['gate'+str(i)] = 0.003
    thres['readout'+ str(i)] = 0.05
thres['cx3_4'] =0.035
thres['cx1_3'] =0.035
thres['cx1_2'] =0.035
thres['cx0_1'] =0.035

metrics = ["accuracy"	,"accuracy-diff",	"accuracy-diff-bad"	,"accuracy-diff-good"	,"accuracy-diff-mid"]
df2 = pd.DataFrame()
df2['timestamp'] = df['timestamp']
df2['gate0'] = df['id0']
df2['gate1'] = df['id1']
df2['gate2'] = df['id2']
df2['gate3'] = df['id3']
df2['gate4'] = df['id4']

df2[[ 'cx1_3', 'cx1_2', 'cx0_1','cx3_4']]=df[[ 'cx1_3', 'cx1_2', 'cx0_1','cx3_4']]
df2[['readout0','readout1','readout2','readout3']] = df[['readout_error_q0','readout_error_q1','readout_error_q2','readout_error_q3']]
acc_corr_dict = {}
print(df2)


acc_corr_dict = {k: v for k, v in sorted(acc_corr_dict.items(), key=lambda item: item[1])}
print(acc_corr_dict)

for key in keys:
    df2[key + '_outlier'] = 0
    df2[key + '_outlier'].loc[df2[key]>thres[key]] =1

for key in keys2:
    df2[key+ '_diff'] =0
    for i in range(len(df2)):
        df2[key+ '_diff'].loc[i] =  df2[key].loc[i]- df2[key].loc[0]

df2['diff_sum'] =0
for i in range(len(df2)):
    for key in ['gate0','gate1','gate2','gate3']:
        if df2[key+ '_diff'].loc[i] >0:
            df2['diff_sum'].loc[i] += abs(df2[key+ '_diff'].loc[i]) 
    for key in [ 'cx1_3', 'cx1_2', 'cx0_1']:
        if df2[key+ '_diff'].loc[i] >0:
            df2['diff_sum'].loc[i] += abs(df2[key+ '_diff'].loc[i])
        
class DecisionMaker:
    def __init__(self) -> None:
        self.kinds = ['error','noisy','shift']
        self.error_kinds = {'qubit_error':0,'cx_error':0,'readout_error':0}
        self.noisy_kinds = {'gate_noisy':0,'cx_noisy':0,'readout_noisy':0}
        self.shift_kinds = {'gate_shift':0,'cx_shift':0,'readout_shift':0}
        self.prob_init = [0,0,0,0]
        self.cx_prob_init = {'cx1_3':0, 'cx1_2':0, 'cx0_1':0}


    def process(self,data):
        result =dict()
        result['error_kinds'] = copy.deepcopy(self.error_kinds)
        result['noisy_kinds'] = copy.deepcopy(self.noisy_kinds)
        result['shift_kinds'] = copy.deepcopy(self.shift_kinds)

        qubit_error_prob = copy.deepcopy(self.prob_init)
        cx_error_prob = copy.deepcopy(self.cx_prob_init)
        # readout_error_prob = copy.deepcopy(self.prob_init)

        #error kinds
        #qubit error
        for i in range(len(self.prob_init)):
            if data['gate{}_outlier'.format(i)] == 1:
                qubit_error_prob[i]+=1.1

        for key  in self.cx_prob_init.keys():
            if data['{}_outlier'.format(key)]==1:
                qubit_error_prob[int(key[2])]+=0.6
                qubit_error_prob[int(key[4])]+=0.6

        #cx error
        for key in self.cx_prob_init.keys():
            if data[key] >0.8:
                cx_error_prob[key] =1.1
        #readout error
        # for i in range(len(self.prob_init)):
        #     if data['readout{}'.format(i)] > 0.5:
        #         readout_error_prob[i] =1.1

        # error summerize
        qubit_error_prob = np.array(qubit_error_prob)
        if (qubit_error_prob >1.0).sum() >= 1:
            result['error_kinds']['qubit_error']=1
            result['qubit_error_prob'] = qubit_error_prob

        cx_error_prob = np.array(list(cx_error_prob.values()))
        if (cx_error_prob >1.0).sum() >= 1:
            result['error_kinds']['cx_error']=1
            result['cx_error_prob'] = cx_error_prob
            
        # readout_error_prob = np.array(readout_error_prob)
        # if (readout_error_prob >1.0).sum() >= 1:
        #     result['error_kinds']['readout_error'] = 1
        #     result['readout_error_prob']= readout_error_prob
        
        gate_noisy_prob = copy.deepcopy( self.prob_init)
        cx_noisy_prob = copy.deepcopy(self.cx_prob_init)
        # readout_noisy_prob = copy.deepcopy(self.prob_init)
        gate_shift_prob = copy.deepcopy(self.prob_init)
        cx_shift_prob = copy.deepcopy(self.cx_prob_init)
        # readout_shift_prob = copy.deepcopy(self.prob_init)

        #noisy/shift kinds
        #qubit noisy/shift
        for i in range(len(self.prob_init)):
            if data['gate{}_diff'.format(i)] > 0.002:
                gate_noisy_prob[i] +=1.1
            elif data['gate{}_diff'.format(i)] < -0.002:
                gate_shift_prob[i] +=1.1


        #cx noisy/shift
        for key in self.cx_prob_init.keys():
            if data[key+'_diff'] >0.02:
                cx_noisy_prob[key] =1.1
            elif data[key+'_diff'] <-0.02:
                cx_shift_prob[key] =1.1
        #readout noisy/shift
        # for i in range(len(self.prob_init)):
        #     if data['readout{}_diff'.format(i)] > 0.04:
        #         readout_noisy_prob[i] =1.1
        #     elif data['readout{}_diff'.format(i)] < -0.04:
        #         readout_shift_prob[i] =1.1

        # noisy/shift summerize
        gate_noisy_prob = np.array(gate_noisy_prob)
        if (gate_noisy_prob >1.0).sum() >= 1:
            result['noisy_kinds']['gate_noisy']=1
            result['gate_noisy_prob'] = gate_noisy_prob

        gate_shift_prob = np.array(gate_shift_prob)
        if (gate_shift_prob >1.0).sum() >= 1:
            result['shift_kinds']['gate_shift']=1
            result['gate_shift_prob'] = gate_shift_prob

        cx_noisy_prob = np.array(list(cx_noisy_prob.values()))
        if (cx_noisy_prob >1.0).sum() >= 1:
            result['noisy_kinds']['cx_noisy']=1
            result['cx_noisy_prob'] = cx_noisy_prob

        cx_shift_prob = np.array(list(cx_shift_prob.values()))
        if (cx_shift_prob >1.0).sum() >= 1:
            result['shift_kinds']['cx_shift']=1
            result['cx_shift_prob'] = cx_shift_prob


        # readout_noisy_prob = np.array(readout_noisy_prob)
        # if (readout_noisy_prob >1.0).sum() >= 1:
        #     result['noisy_kinds']['readout_noisy']=1
        #     result['readout_noisy_prob'] = readout_noisy_prob

        # readout_shift_prob = np.array(readout_shift_prob)
        # if (readout_shift_prob >1.0).sum() >= 1:
        #     result['shift_kinds']['readout_shift']=1
        #     result['readout_shift_prob'] = readout_shift_prob
        return result

decision_maker = DecisionMaker()
df2['qubit_error'] = 0
df2['gate_noisy'] =0
df2['cx_noisy'] = 0
# df2['readout_noisy'] =0

for i in range(len(df2)):
    result= decision_maker.process(df2.loc[i])
    df2['qubit_error'].loc[i] = result['error_kinds']['qubit_error']
    df2['gate_noisy'].loc[i]  = result['noisy_kinds']['gate_noisy']
    df2['cx_noisy'].loc[i]  = result['noisy_kinds']['cx_noisy']
    # df2['readout_noisy'].loc[i] = result['noisy_kinds']['readout_noisy']

df2['accuracy'] = df["accuracy-diff-good"]


df2.to_excel(path_result)



# import matplotlib.pyplot as plt

# for key in keys2:
#     key =key +'_diff'
#     plt.scatter(df2[key],df2['accuracy'])
#     plt.savefig(output_path+key +'.jpg' )
#     plt.cla()
# plt.show()