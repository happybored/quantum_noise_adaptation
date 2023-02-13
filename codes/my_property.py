from qiskit.providers.models import BackendProperties
from qiskit import IBMQ
import datetime

class IBMMachinePropertiesFactory:
    def __init__(self, provider):
        self.provider = provider

    @property
    def machines(self):
        machines = []
        backends = self.provider.backends()
        for backend_i in backends:
            # skip simulators
            if (backend_i.configuration().backend_name) == ('ibmq_qasm_simulator') or (backend_i.configuration().backend_name) == ('simulator_statevector')\
                    or (backend_i.configuration().backend_name) == ('simulator_mps') or (backend_i.configuration().backend_name) == ('simulator_extended_stabilizer') \
                    or (backend_i.configuration().backend_name) == ('simulator_stabilizer') or (backend_i.configuration().backend_name) == ('ibmq_armonk'):
                continue
            machines.append(backend_i.configuration().backend_name)
        return machines


    def query_backend_info(self, backend_name, query_date=None):
        """Query info about a given backend at a specific time. Returns a tuple (datetime, info)."""
        backend = self.provider.get_backend(backend_name)
        config = backend.configuration()
        props = backend.properties(datetime=query_date)
        if props is None:
            return None, None
        return config,props,backend

class CalibDictionary(object):
    def __init__(self):
        self.model_dict = dict()
        self.thres = 0.03
        self.date_format =  "%Y-%m-%d"

    def calculate_dis(self,calib_data,mid_calib_data):
        dis = 0
        keys = ['id0','id1','id2','id3', 'cx1_3', 'cx1_2','cx0_1']
        for key in keys:
            dis += abs(calib_data[key] -mid_calib_data[key])
        return dis

    def get_nearest_model(self,calib_data):
        near_dis =1000
        nearset_model = ''
        for model, calib in self.model_dict.items():
            dis = self.calculate_dis(calib_data,calib)
            if dis <near_dis:
                near_dis = dis
                nearset_model = model
        return nearset_model,near_dis

    def process(self,calib_data):
        # print(calib_data)
        calib_time = calib_data['timestamp'][0] + datetime.timedelta(days=1)

        now_query_time = calib_time.strftime(self.date_format)
        train_situation = dict()
        if len(self.model_dict) == 0:
            self.model_dict[now_query_time] = calib_data
            train_situation['query_model'] =now_query_time
            train_situation['train_flag'] = True
            # print(calib_data)
            return train_situation          
        near_query_time,dis = self.get_nearest_model(calib_data)
        if dis < self.thres: # 
            train_situation['query_model'] =near_query_time
            train_situation['train_flag'] = False
            return train_situation
        else:
            # print(calib_data)
            self.model_dict[now_query_time] = calib_data
            train_situation['query_model'] =now_query_time
            train_situation['train_flag'] = True
            return train_situation





class MyBackendNoise(object):
    def __init__(self,backend_name,timestamp= None):
        self.__property = None
        self.__noise = None
        self.__backend_name = backend_name
        self.__timestamp = timestamp

    def __create__property(self):
        if self.__property == None:
            provider = IBMQ.load_account()
            backend = provider.get_backend(self.__backend_name)
            props = backend.properties(datetime=self.__timestamp)
            self.__property = props

    def set_property(self,property):
        self.__property = property
        self.__property2noise()

    def set_noise(self,noise):
        self.__noise = noise
        self.__noise2property()

    def __create_noise_version(self):
        noise_version = {
        'backend_name':[self.__backend_name],
        'timestamp': [self.__property.last_update_date]}
        return noise_version
    
    def __property2noise(self):
        noise_version = self.__create_noise_version()
        for gate_i in self.__property.gates:
            gate_name = gate_i.gate
            if gate_name not in ['reset']:
                noise_version[gate_i.name] = gate_i.parameters[0].value
        property_dict =  self.__property.to_dict()
        i = 0
        for qubit in property_dict['qubits']:
            for info in qubit:
                if info['name'] in ['T1','T2','frequency','readout_error','prob_meas0_prep1','prob_meas1_prep0']:
                    noise_version[info['name']+'_q'+str(i)]= info['value']
            i = i+1
        self.__noise = noise_version
        

    def __noise2property(self):
        self.__create__property()
        property_dict = self.__property.to_dict()
        i = 0
        for qubit in property_dict['qubits']:
            for info in qubit:
                if info['name'] in ['T1','T2','frequency','readout_error','prob_meas0_prep1','prob_meas1_prep0']:
                    info['value'] = self.__noise[info['name']+'_q'+str(i)]
            i = i+1
        for gate_i in property_dict['gates']:
            gate_name = gate_i["gate"]
            if gate_name not in ['reset']:
                gate_i["parameters"][0]["value"] = self.__noise[gate_i["name"]]
        self.__property = BackendProperties.from_dict(property_dict)
        


    def get_property(self):
        return self.__property

    def get_noise(self):
        return self.__noise