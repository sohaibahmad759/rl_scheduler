import copy
import logging
import random
import requests
import string
import os
import time
import pprint
import numpy as np
import pandas as pd
from hashlib import new
from pyexpat import model
from core.common import Event, EventType
from core.executor import Executor, AccType
from core.exceptions import SimulatorException

MAX_CLOCK_VALUE = 1439934
MAX_CLOCK_VALUE = 0
# If the simulator clock is within this threshold of the starting time of
# the last request in queue for a given executor, it will refill the event
# queue
QUEUE_REFILL_THRESHOLD = 2000


class Simulator:
    def __init__(self, job_sched_algo, logging_level, trace_path=None,
                 mode='training', max_acc_per_type=0, predictors_max=[10, 10, 10, 10],
                 n_qos_levels=1, random_runtimes=False, fixed_seed=0, batching=False,
                 model_assignment=None, batching_algo=None, profiling_data=None,
                 allowed_variants_path=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging_level)
        self.logging_level=logging_level

        self.clock = 0
        self.event_queue = []
        self.executors = {}
        self.idx_to_executor = {}
        self.isi_to_idx = {}
        self.failed_requests = 0
        self.total_failed = 0
        self.total_successful = 0
        self.failed_requests_arr = []
        self.successful_requests = 0
        self.successful_requests_arr = []
        self.total_requests_arr = []
        self.mode = mode
        self.completed_requests = 0

        self.max_acc_per_type = max_acc_per_type

        if fixed_seed != 0:
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)

        self.job_sched_algo = job_sched_algo
        self.n_qos_levels = n_qos_levels

        self.test_stat = 0

        self.store_file_pointers = True
        self.requests_added = 0

        self.total_accuracy = 0

        self.accelerator_types = ['CPU', 'GPU_AMPERE', 'VPU', 'GPU_PASCAL']

        self.cpu_runtimes = {}
        self.gpu_runtimes = {}
        self.vpu_runtimes = {}
        self.fpga_runtimes = {}

        self.cpu_loadtimes = {}
        self.gpu_loadtimes = {}
        self.vpu_loadtimes = {}
        self.fpga_loadtimes = {}

        self.model_variants = {}
        self.model_variant_runtimes = {}
        self.model_variant_loadtimes = {}
        self.model_variant_accuracies = {}

        self.cpu_variant_runtimes = {}
        self.gpu_variant_runtimes = {}
        self.vpu_variant_runtimes = {}
        self.fpga_variant_runtimes = {}

        self.requests_per_model = {}
        self.failed_per_model = {}
        self.successful_per_model = {}

        self.slo_timeouts = {'total': 0, 'succeeded': 0, 'timeouts': 0}

        self.accuracy_per_model = {}

        self.trace_files = {}
        self.trace_file_finished = set()
        self.last_request_starting = {}

        self.accuracy_logfile = open('logs/log_ilp_accuracy.txt', mode='w')
        self.latency_logfilename = os.path.join('logs', 'latency',
                                                f'{time.time()}_{model_assignment}.csv')
        self.latency_logfile = open(self.latency_logfilename, mode='w')

        self.predictors_max = predictors_max
        # self.available_predictors = np.tile(copy.deepcopy(predictors_max), self.n_qos_levels)
        self.available_predictors = copy.deepcopy(predictors_max)

        self.batching_enabled = batching
        self.batching_algo = batching_algo
        self.largest_batch_sizes = {}
        self.slo_dict = {}
        self.allowed_batch_sizes = [1, 2, 4, 8]
        self.profiled_filename = profiling_data

        self.model_assignment = model_assignment

        self.infaas_slack = 0.95

        idx = 0

        if trace_path is None:
            raise SimulatorException(f'No trace file provided. trace_path: {trace_path}')
            
        self.log.info(f'Reading trace files from: {trace_path}')
        trace_files = sorted(os.listdir(trace_path))
        for file in trace_files:
            filename = file.split('/')[-1]
            if len(filename.split('.')) == 1:
                # it is a directory
                continue
            isi_name, extension = filename.split('.')
            if not extension == 'txt':
                continue
            self.log.info('Filename: ' + file)

            variant_list_filename = os.path.join(allowed_variants_path, isi_name)
            self.log.info('Variant list filename:' + variant_list_filename)

            model_variant_list = self.read_variants_from_file(variant_list_filename)
            self.set_model_variants(isi_name, model_variant_list)

            self.initialize_model_variant_loadtimes(isi_name)

            self.add_executor(isi_name, self.job_sched_algo, runtimes={},
                              model_variant_runtimes={}, model_variant_loadtimes={},
                              max_acc_per_type=self.max_acc_per_type, simulator=self)
            self.idx_to_executor[idx] = isi_name
            self.isi_to_idx[isi_name] = idx
            self.failed_requests_arr.append(0)
            self.successful_requests_arr.append(0)
            self.total_requests_arr.append(0)
            idx += 1

            if self.store_file_pointers:
                readfile = open(os.path.join(trace_path, file), mode='r')
                self.trace_files[isi_name] = readfile
                self.add_requests_from_trace_pointer(
                    isi_name, readfile, read_until=10000)
            else:
                start = time.time()
                self.add_requests_from_trace(
                    isi_name, os.path.join(trace_path, file))
                end = time.time()
                self.log.debug(
                    'Time to add trace file: {} seconds'.format(end - start))
        
        self.initialize_model_variant_runtimes()

        self.runtimes = {1: self.cpu_runtimes, 2: self.gpu_runtimes,
                         3: self.vpu_runtimes, 4: self.fpga_runtimes}

        self.loadtimes = {1: self.cpu_loadtimes, 2: self.gpu_loadtimes,
                          3: self.vpu_loadtimes, 4: self.fpga_loadtimes}

        self.set_executor_runtimes()
        self.set_executor_loadtimes()
        self.log.info(f'Model variant accuracies: {self.model_variant_accuracies}')

        self.set_executor_model_variants()

        self.set_executor_variant_accuracies()
        self.set_executor_variant_runtimes()
        self.set_executor_variant_loadtimes()

        self.set_largest_batch_sizes()

        self.qos_stats = np.zeros((len(self.executors), n_qos_levels * 2))

        if not (mode == 'training' or mode == 'debugging'):
            # if the RL agent is being trained, we don't need to invoke scheduling microservice
            self.insert_event(0, EventType.SCHEDULING,
                              'invoke scheduling agent')

        # how frequently the scheduling agent is invoked (in milliseconds)
        self.sched_interval = 200

        self.sched_agent_uri = 'http://127.0.0.1:8000'

    
    def set_executor_runtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_runtimes(self.runtimes)

    
    def set_executor_model_variants(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_model_variants(self.model_variants)


    def set_executor_variant_accuracies(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_variant_accuracies(self.model_variant_accuracies)


    def set_executor_variant_loadtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_variant_loadtimes(self.model_variant_loadtimes)

    
    def set_executor_variant_runtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_variant_runtimes(self.model_variant_runtimes)


    def set_executor_loadtimes(self):
        for idx in self.idx_to_executor:
            isi_name = self.idx_to_executor[idx]
            self.executors[isi_name].set_loadtimes(self.loadtimes)

    
    def read_variants_from_file(self, filename):
        model_variants = []
        isi_name = filename.split('/')[-1]
        print(f'filename: {filename}')
        if os.path.exists(filename):
            with open(filename, mode='r') as rf:
                for line in rf.readlines():
                    model_variant = line.split(',')[0]
                    accuracy = float(line.rstrip('\n').split(',')[1])

                    model_variants.append(model_variant)
                    self.model_variant_accuracies[(isi_name, model_variant)] = accuracy
        else:
            raise SimulatorException(f'path not found: {filename}')
        return model_variants


    def reset(self):
        print('Resetting simulator')
        return


    def add_requests_from_trace(self, isi_name, file):
        ''' Completely read trace file into memory.
        '''
        with open(file, mode='r') as readfile:
            for line in readfile:
                request_description = line.rstrip('\n').split(',')
                start_time = int(request_description[0])
                # if qos_level is defined, use that. otherwise use qos_level 0 by default
                if len(request_description) >= 2:
                    try:
                        qos_level = int(request_description[1])
                    except ValueError:
                        accuracy = float(request_description[1])
                        qos_level = 0
                else:
                    qos_level = 0
                # also, if n_qos_levels is 1, ignore qos info from trace
                if self.n_qos_levels == 1:
                    qos_level = 0

                if len(request_description) >= 3:
                    deadline = request_description[2]
                else:
                    deadline = 1000

                if len(request_description) >= 4:
                    accuracy = float(request_description[3])
                else:
                    accuracy = 100.0
                
                self.insert_event(start_time, EventType.START_REQUEST, isi_name, runtime=None,
                                  deadline=deadline, qos_level=qos_level, accuracy=accuracy)
        return

    def add_requests_from_trace_pointer(self, isi_name, readfile, read_until=500):
        ''' Partially read trace file into memory and store pointer.
        '''
        # read_until = 500
        # readfile.readable()
        while True:
            line = readfile.readline()
            if not line:
                # self.trace_files[readfile] = -1
                return True
            if line.strip()[0] == '#':
                continue  # Skip comment lines in file
            request_description = line.rstrip('\n').split(',')
            start_time = int(request_description[0])
            # if qos_level is defined, use that. otherwise use qos_level 0 by default
            if len(request_description) >= 2:
                try:
                    qos_level = int(request_description[1])
                except ValueError:
                    accuracy = float(request_description[1])
                    qos_level = 0
            else:
                qos_level = 0
            # also, if n_qos_levels is 1, ignore qos info from trace
            if self.n_qos_levels == 1:
                qos_level = 0

            if len(request_description) >= 3:
                deadline = float(request_description[2])
                self.slo_dict[isi_name] = deadline
                # print(deadline)
            else:
                deadline = 1000

            if len(request_description) >= 4:
                accuracy = float(request_description[3])
            else:
                accuracy = 100.0

            self.insert_event(start_time, EventType.START_REQUEST, isi_name, runtime=None,
                              deadline=deadline, qos_level=qos_level, accuracy=accuracy)
            self.requests_added += 1
            if start_time >= read_until:
                break
            # print(self.requests_added)
            # We want to keep track of where we stopped adding requests for
            # each file, so that when the clock gets closer to that time,
            # we can refill the event queue
            self.last_request_starting[isi_name] = start_time
        return False

    def set_model_variants(self, isi_name, model_variant_list):
        self.model_variants[isi_name] = model_variant_list

        self.requests_per_model[isi_name] = 0
        self.failed_per_model[isi_name] = 0
        self.successful_per_model[isi_name] = 0

        self.accuracy_per_model[isi_name] = []

    def initialize_dummy_runtimes(self, isi_name, random_runtimes=False):
        for qos_level in range(self.n_qos_levels):
            for batch_size in self.allowed_batch_sizes:
                if random_runtimes:
                    self.cpu_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                    self.gpu_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                    self.vpu_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                    self.fpga_runtimes[isi_name,
                                    qos_level,
                                    batch_size] = random.randint(20, 100)
                else:
                    self.cpu_runtimes[isi_name, qos_level, batch_size] = 50
                    self.gpu_runtimes[isi_name, qos_level, batch_size] = 25
                    self.vpu_runtimes[isi_name, qos_level, batch_size] = 35
                    self.fpga_runtimes[isi_name, qos_level, batch_size] = 30
                    # self.cpu_runtimes[isi_name, qos_level] = random.randint(25, 100)
                    # self.gpu_runtimes[isi_name, qos_level] = random.randint(25, 100)
                    # self.vpu_runtimes[isi_name, qos_level] = random.randint(25, 100)
                    # self.fpga_runtimes[isi_name, qos_level] = random.randint(25, 100)
                # print(self.cpu_runtimes)
                # print(self.gpu_runtimes)
                # print(self.vpu_runtimes)
                # print(self.fpga_runtimes)
        
        # Initialize the runtimes per model variant as well
        self.initialize_dummy_model_variant_runtimes(isi_name, random_runtimes=random_runtimes)
        return

    def get_thput_accuracy_per_model(self):
        requests = []
        failed = []
        accuracy = []
        successful = []
        for model in self.requests_per_model:
            requests.append(self.requests_per_model[model])
            self.requests_per_model[model] = 0
            
            failed.append(self.failed_per_model[model])
            self.failed_per_model[model] = 0

            total_accuracy = sum(self.accuracy_per_model[model])
            accuracy.append(total_accuracy)
            self.accuracy_per_model[model] = []

            successful.append(self.successful_per_model[model])
            self.successful_per_model[model] = 0

        return requests, failed, accuracy, successful

    def initialize_model_variant_loadtimes(self, isi_name):
        model_variant_list = self.model_variants[isi_name]
        for model_variant in model_variant_list:
            self.model_variant_loadtimes[(isi_name, model_variant)] = 0

    def initialize_model_variant_runtimes(self):
        profiled = self.import_profiled_data()
        model_variants = profiled['model_variant']
        acc_types = profiled['acc_type']
        batch_sizes = profiled['batch_size']
        latencies = profiled['latency']
        isi_list = []
        for idx in range(profiled.shape[0]):
            model_variant = model_variants[idx]
            if model_variant not in sum(self.model_variants.values(), []):
                # the model variant is profiled but not in allowed_variants from config
                continue
            isi_name = self.get_isi_from_variant_name(model_variant)
            if isi_name not in isi_list:
                isi_list.append(isi_name)
            acc_type = acc_types[idx]
            batch_size = batch_sizes[idx]
            latency = latencies[idx]

            if batch_size not in self.allowed_batch_sizes:
                continue

            # Since the profiled data does not have VPUs, we make the following
            # assumption to reconcile this data with the simulator experiment
            # setting: Assume that VPUs behave the same as CPUs. So effectively
            # we are left with twice the number of CPUs and 0 VPUs
            tuple = (isi_name, model_variant, batch_size)
            if acc_type == 'CPU':
                self.cpu_variant_runtimes[tuple] = latency
                self.vpu_variant_runtimes[tuple] = latency
            elif acc_type == 'GPU_AMPERE':
                self.gpu_variant_runtimes[tuple] = latency
            elif acc_type == 'GPU_PASCAL':
                self.fpga_variant_runtimes[tuple] = latency

            self.log.debug(f'({isi_name}, {model_variant}, {batch_size}): {latency}, {acc_type}')
        
        # If there are unfilled entries that have not been profiled, set their
        # latencies to be effectively infinite so we don't use them
        for isi_name in isi_list:
            model_variant_list = self.model_variants[isi_name]
            for model_variant in model_variant_list:
                for batch_size in self.allowed_batch_sizes:
                    tuple = (isi_name, model_variant, batch_size)

                    if tuple not in self.cpu_variant_runtimes:
                        # self.cpu_variant_runtimes[tuple] = batch_size*100
                        # self.log.error(f'{tuple} not found in cpu variant runtimes')
                        raise SimulatorException(f'{tuple} not found in cpu variant runtimes')
                    if tuple not in self.gpu_variant_runtimes:
                        # self.gpu_variant_runtimes[tuple] = batch_size*100
                        # self.log.error(f'{tuple} not found in gpu_ampere variant runtimes')
                        raise SimulatorException(f'{tuple} not found in gpu_ampere variant runtimes')
                    if tuple not in self.vpu_variant_runtimes:
                        # self.vpu_variant_runtimes[tuple] = batch_size*100
                        # print(f'{tuple} not found in vpu variant runtimes')
                        raise SimulatorException(f'{tuple} not found in vpu variant runtimes')
                    if tuple not in self.fpga_variant_runtimes:
                        # self.fpga_variant_runtimes[tuple] = batch_size*100
                        self.log.error(f'{tuple} not found in gpu_pascal variant runtimes')
                        # raise SimulatorException(f'{tuple} not found in gpu_pascal variant runtimes')

        self.log.debug(f'Total profiled entries: {profiled.shape[0]}')
        # self.log.debug(f'Total expected entries: {(22*4*3)}')

        self.model_variant_runtimes = {1: self.cpu_variant_runtimes, 2: self.gpu_variant_runtimes,
                                       3: self.vpu_variant_runtimes, 4: self.fpga_variant_runtimes}
        return

    def set_largest_batch_sizes(self):
        ''' Accesses the profiled data to establish the maximum batch size that
        does not violate latency SLO for each (variant, accelerator) pair
        '''
        max_batch_size_dict = {}
        
        for isi_name in self.model_variants:
            model_variants = self.model_variants[isi_name]
            for model_variant in model_variants:
                # isi_name = self.simulator.get_isi_from_variant_name(model_variant)
                for acc_type in self.accelerator_types:
                    acc_latencies = {}
                    if acc_type == 'CPU':
                        acc_latencies = self.model_variant_runtimes[1]
                    elif acc_type == 'GPU_AMPERE':
                        acc_latencies = self.model_variant_runtimes[2]
                    elif acc_type == 'VPU':
                        acc_latencies = self.model_variant_runtimes[3]
                    elif acc_type == 'GPU_PASCAL':
                        acc_latencies = self.model_variant_runtimes[4]

                    max_batch_size = 0
                    for batch_size in self.allowed_batch_sizes:
                        if (isi_name, model_variant, batch_size) not in acc_latencies:
                            acc_latencies[(isi_name, model_variant, batch_size)] = np.inf
                        latency = acc_latencies[(isi_name, model_variant, batch_size)]

                        # self.log.error(f'slo_dict: {self.slo_dict}')
                        # if 'gpt' in model_variant:
                        #     self.log.error(f'variant: {model_variant}, batch_size: {batch_size}, '
                        #                    f'latency: {latency}, slo/2: {(self.slo_dict[isi_name]/2)}, '
                        #                    f'latency < slo/2: {latency < self.slo_dict[isi_name]/2}, '
                        #                    f'acc_type: {acc_type}')
                        #     time.sleep(1)
                        if batch_size > max_batch_size and latency < self.slo_dict[isi_name] / 2:
                            max_batch_size = batch_size

                    max_batch_size_dict[(acc_type, model_variant)] = max_batch_size
                    self.log.debug(f'({acc_type}, {model_variant}): {max_batch_size}')
        self.log.debug(f'len(largest_batch_sizes): {len(max_batch_size_dict)}')
        self.log.info(f'largest_batch_sizes:')
        pprint.pprint(max_batch_size_dict)
        # raise SimulatorException(f'if latency data is not present (OOM while profiling), '
        #                          f'it is assuming largest batch size (8) for some reason')
        self.largest_batch_sizes = max_batch_size_dict
        return
    
    def get_largest_batch_sizes(self):
        return self.largest_batch_sizes
    
    def initialize_dummy_model_variant_runtimes(self, isi_name, random_runtimes=False):
        model_variant_list = self.model_variants[isi_name]
        for model_variant in model_variant_list:
            for batch_size in self.allowed_batch_sizes:
                if random_runtimes:
                    self.cpu_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                    self.gpu_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                    self.vpu_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                    self.fpga_variant_runtimes[(isi_name, model_variant, batch_size)] = random.randint(20, 100)
                else:
                    self.cpu_variant_runtimes[(isi_name, model_variant, batch_size)] = 50
                    self.gpu_variant_runtimes[(isi_name, model_variant, batch_size)] = 25
                    self.vpu_variant_runtimes[(isi_name, model_variant, batch_size)] = 35
                    self.fpga_variant_runtimes[(isi_name, model_variant, batch_size)] = 30
        
        self.model_variant_runtimes = {1: self.cpu_variant_runtimes, 2: self.gpu_variant_runtimes,
                                        3: self.vpu_variant_runtimes, 4: self.fpga_variant_runtimes}
        return

    def replace_profiled_strings(self, element):
        if type(element) is not str:
            return element
        elif element == 'onnxruntime_gpu_ampere':
            return 'GPU_AMPERE'
        elif element == 'onnxruntime_gpu_pascal':
            return 'GPU_PASCAL'
        elif element == 'onnxruntime_cpu':
            return 'CPU'
        elif '.onnx' in element:
            return element.rstrip('.onnx').split('_')[0]
        else:
            return element

    def get_isi_from_variant_name(self, model_variant):
        ''' Gets ISI name from the variant name based on filename
        '''
        # if '_v1' in model_variant:
        #     model_variant = model_variant.split('_')[0]
            
        for element in self.model_variant_accuracies:
            (isi_name, variant) = element
            if model_variant == variant:
                return isi_name

        raise SimulatorException(f'isi_name not found. model variant: {model_variant}')

    def import_profiled_data(self):
        ''' Reads and parses the profiled latency data
        '''
        profiled = pd.read_csv(self.profiled_filename)
        profiled = profiled.rename({'Model': 'model_variant', 'Accel': 'acc_type',
                                    'batchsize': 'batch_size', 'avg_latency(ms)': 'latency'},
                                    axis='columns')
        profiled = profiled.applymap(self.replace_profiled_strings)
        
        pd.set_option('display.max_columns', None)
        self.log.info(f'profiled data: {profiled}')
        
        return profiled

    def initialize_loadtimes(self, isi_name):
        self.initialize_model_variant_loadtimes(isi_name)
        for qos_level in range(self.n_qos_levels):
            self.cpu_loadtimes[isi_name, qos_level] = 0
            self.gpu_loadtimes[isi_name, qos_level] = 0
            self.vpu_loadtimes[isi_name, qos_level] = 0
            self.fpga_loadtimes[isi_name, qos_level] = 0
        return

    def trigger_infaas_upscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_upscaling()

    def trigger_infaas_v2_upscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_v2_upscaling()

    def trigger_infaas_downscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_downscaling()

    def trigger_infaas_v2_downscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_v2_downscaling()

    def get_runtimes(self, isi_index):
        runtimes = []
        for qos_level in range(self.n_qos_levels):
            isi_name = self.idx_to_executor[isi_index]
            runtimes.append(self.cpu_runtimes[isi_name, qos_level])
            runtimes.append(self.gpu_runtimes[isi_name, qos_level])
            runtimes.append(self.vpu_runtimes[isi_name, qos_level])
            runtimes.append(self.fpga_runtimes[isi_name, qos_level])
        # print(runtimes)
        return runtimes

    # until -1 means till the end, any other number would specify the clock time

    def simulate_until(self, until=-1):
        '''
        Run the simulator until the specified clock time.
        If -1 is specified, run till the end.
        '''
        while len(self.event_queue) > 0 and (self.event_queue[0].start_time <= until or until == -1):
            current_event = self.event_queue.pop(0)
            self.clock = current_event.start_time
            self.process(current_event, self.clock)

            for isi in self.last_request_starting:
                last_request_starting = self.last_request_starting[isi]
                if self.clock + QUEUE_REFILL_THRESHOLD >= last_request_starting:
                    # in case we are loading file pointers instead of entire file
                    if self.store_file_pointers:
                        finished = self.refill_event_queue(isi)
                        # if finished:
                        #     break
        # print(f'ending simulate_until. until: {until}, queue length: {len(self.event_queue)}, '
        #       f'first event start time: {self.event_queue[0].start_time}')
        # time.sleep(5)
        return

    def refill_event_queue(self, isi=None):
        # self.log.debug(f'simulator clock: {self.clock}')
        if len(self.trace_files) == 0:
            # we are in a deepcopied simulator instance
            return False
        # if len(self.event_queue) < 10000:
        finished = True

        if isi is None:
            trace_files = self.trace_files
        else:
            trace_files = {isi: self.trace_files[isi]}

        for isi_name in trace_files:
            if isi_name in self.trace_file_finished:
                continue
            readfile = self.trace_files[isi_name]
            # print('Adding requests from {}'.format(readfile))
            file_finished = self.add_requests_from_trace_pointer(isi_name, readfile,
                                                                read_until=self.clock+10000)
            if file_finished:
                self.trace_file_finished.add(isi_name)
                self.log.info('Trace file {} finished'.format(isi_name))
                time.sleep(1)
            finished = finished and file_finished
        return finished
        # else:
        #     print(f'refill_event_queue: unresolved simulator issue')
        #     time.sleep(50)
        # return False

    def is_done(self):
        '''
        Returns true if the request trace has finished, false otherwise.
        '''
        if len(self.event_queue) == 0 or (MAX_CLOCK_VALUE > 0 and self.clock > MAX_CLOCK_VALUE):
            return True
        else:
            return False

    def simulate_requests(self, requests=10):
        '''
        Run the simulator until a given number of requests (instead of a clock time).
        '''
        while len(self.event_queue) > 0 and requests > 0:
            current_event = self.event_queue.pop(0)
            # we only want to process requests and skip scheduling events
            if current_event.type == EventType.SCHEDULING:
                self.reset_request_count()
                continue
            self.clock = current_event.start_time
            self.process(current_event, self.clock)
            requests -= 1

            if self.store_file_pointers:
                finished = self.refill_event_queue()
                if finished:
                    print('oops')
                    break
        return

    def reset_request_count(self):
        for i in range(len(self.total_requests_arr)):
            self.total_requests_arr[i] = 0
            self.failed_requests_arr[i] = 0
            self.failed_requests = 0
            self.successful_requests = 0
            self.successful_requests_arr[i] = 0
        self.qos_stats = np.zeros((len(self.executors), self.n_qos_levels * 2))

    def get_total_requests_arr(self):
        return self.total_requests_arr

    def get_qos_stats(self):
        return self.qos_stats

    def get_failed_requests_arr(self):
        return self.failed_requests_arr

    def get_failed_requests(self):
        return self.failed_requests

    def get_successful_requests_arr(self):
        return self.successful_requests_arr

    def get_successful_requests(self):
        return self.successful_requests

    # def reset_failed_requests(self):
    #     self.failed_requests = 0
        # self.qos_stats = np.zeros((len(self.executors), self.n_qos_levels))

    def print_assignment(self):
        self.log.debug('Printing assignment in simulator...')
        for isi in self.executors:
            assignment = self.executors[isi].num_predictor_types
            self.log.debug(
                'Executor {} has assignment: {}'.format(isi, assignment))
        return

    def apply_ilp_solution(self, required_predictors, canary_dict, ilp_x):
        ''' Takes in two dictionaries, (i) required_predictors to indicate
        model assignment, and (ii) canary_dict to define the canary routing
        table for the given model assignment
        '''
        self.apply_predictor_dict(required_predictors)
        self.apply_canary_dict(canary_dict, ilp_x)
        return

    def apply_predictor_dict(self, required_predictors={}):
        """ Takes in a dictionary 'required_predictors' with
        key: (model_variant, accelerator_type)
        value: number of predictors
        And applies the assignment to current system
        """
        if len(required_predictors) == 0:
            raise SimulatorException('No required predictors passed')
            time.sleep(5)
            return

        self.log.debug(f'apply_predictor_dict: required_predictors: {required_predictors}')

        ilp_to_acc_type = {}
        ilp_to_acc_type['CPU'] = 1
        ilp_to_acc_type['GPU_AMPERE'] = 2
        ilp_to_acc_type['VPU'] = 3
        ilp_to_acc_type['GPU_PASCAL'] = 4

        variant_to_executor = {}

        # Build a dictionary 'existing_predictors'
        existing_predictors = {}
        for ex_key in self.executors:
            executor = self.executors[ex_key]
            # print('executor {}, name: {}'.format(executor, executor.isi))
            # print('executor.predictors:', executor.predictors)
            for pred_list_key in executor.predictors:
                predictor = executor.predictors[pred_list_key]
                model_variant = predictor.variant_name
                acc_type = predictor.acc_type
                
                tuple_key = (model_variant, acc_type)

                if tuple_key in existing_predictors:
                    existing_predictors[tuple_key].append(predictor)
                else:
                    existing_predictors[tuple_key] = [predictor]

            # Build a reverse dictionary to get executor from model variant name
            model_variants = executor.variants_for_this_executor
            for variant in model_variants:
                variant_to_executor[variant] = executor
        # time.sleep(3)

        # total_required = sum(required_predictors.values())
        # total_existing = sum(len(x) for x in existing_predictors.values())
        # print('total_required predictors:', total_required)
        # print('total_existing predictors', total_existing)
        # if (total_required > 40):
        #     time.sleep(10)

        self.log.debug(f'Existing predictors: {existing_predictors}')

        existing_count = {}
        for key in existing_predictors:
            (model_variant, acc_type) = key
            existing_count[(model_variant, AccType(acc_type))] = len(existing_predictors[key])
        # print('existing predictor count:', sorted(existing_count))
        # print('required predictors:', sorted(required_predictors))
        # time.sleep(3)

        self.log.debug(f'Existing count: {existing_count}')

        all_tuple_combinations = self.generate_all_model_acc_keys()

        # Go through each key:
        for tuple_key in all_tuple_combinations:
            # Remove predictors if needed
            (model_variant, accelerator_type) = tuple_key
            acc_type = ilp_to_acc_type[accelerator_type]
            
            existing_key = (model_variant, acc_type)
            existing = len(existing_predictors.get(existing_key, {}))
            required = required_predictors.get(tuple_key, 0)
            self.log.debug(f'tuple_key: {tuple_key}, existing: {existing}, required: {required}')

            while existing > required:
                predictor = existing_predictors[(model_variant, acc_type)].pop()
                id = predictor.id
                executor = predictor.executor
                if (executor.remove_predictor_by_id(id)):
                    self.log.debug(f'removed predictor {id} of type {predictor.acc_type} for '
                          f'model variant {model_variant}')
                    self.available_predictors[acc_type-1] += 1
                else:
                    self.log.debug('no predictor found with id:', id)
                    time.sleep(2)

                existing -= 1

        # Go through each key:
        added = 0
        for tuple_key in required_predictors:
            (model_variant, accelerator_type) = tuple_key
            acc_type = ilp_to_acc_type[accelerator_type]

            required = required_predictors[tuple_key]
            if not((model_variant, acc_type) in existing_predictors):
                existing = 0
            else:
                existing = len(existing_predictors[(model_variant, acc_type)])

            # Add predictors if needed
            while required > existing:
                if (self.available_predictors[acc_type-1]) > 0:
                    executor = variant_to_executor[model_variant]
                    executor.add_predictor(acc_type=AccType(acc_type), variant_name=model_variant)
                    self.available_predictors[acc_type-1] -= 1
                    self.log.debug(f'added predictor of type {acc_type} for model variant '
                                   f'{model_variant} at isi: {executor.isi}, id: {executor.id}')
                    self.log.debug(f'isi {executor.isi} now has following predictors: {executor.predictors}')
                    added += 1
                if (self.available_predictors[acc_type-1]) < 0:
                    raise SimulatorException(f'Available predictors {self.available_predictors[acc_type-1]} for type {acc_type}')
                required -= 1
        self.log.debug('added predictors:', added)
        
        return

    def generate_all_model_acc_keys(self):
        ''' Generates a list of tuples of the format (model variant, acc type)
        that enumerates all possible combinations
        '''
        all_model_variants = list(map(lambda x: x[1], self.model_variants.items()))
        # merge list of lists into one list
        all_model_variants = sum(all_model_variants, [])

        all_acc_types = ['CPU', 'GPU_PASCAL', 'VPU', 'GPU_AMPERE']

        all_tuples = []
        for variant in all_model_variants:
            for acc_type in all_acc_types:
                tuple = (variant, acc_type)
                all_tuples.append(tuple)
        return all_tuples

    def apply_canary_dict(self, canary_dict, ilp_x):
        ''' Takes in d dictionary 'canary_dict' with
        key: (model_variant, isi number)
        value: canary routing percentage
        And applies it to the canary routing table for each ISI (executor)
        '''
        self.log.debug(f'len: {len(canary_dict)}, canary_dict: {canary_dict}')
        if len(canary_dict) == 0:
            self.log.error('apply_canary_dict: No canary dictionary passed')
            time.sleep(5)
            return
        
        routing_table_ijk = {}
        for key1 in canary_dict:
            for key2 in ilp_x:
                (accelerator_1, rtype_1) = key1
                (variant, accelerator_2) = key2
                canary_value = canary_dict[key1]
                if accelerator_1 == accelerator_2:
                    routing_table_ijk[(accelerator_1, variant, rtype_1)] = canary_value
        self.log.debug(f'routing_table_ijk: {routing_table_ijk}')

        for idx in self.idx_to_executor:
            isi = self.idx_to_executor[idx]
            executor = self.executors[isi]

            executor_routing_table = dict(filter(lambda x: x[0][2] == idx, routing_table_ijk.items()))
            # just cleaning up from format {('CPU-0', 'roberta-large', 0): canary_pct}
            # to {('CPU', 'roberta-large'): canary_pct}
            executor_routing_table = dict(map(lambda x: ((x[0][0].split('-')[0], x[0][1]), x[1]),
                                              executor_routing_table.items()))
            executor.apply_routing_table(executor_routing_table)
        return

    def null_action(self, action, idx):
        action[0] = idx
        action[1:] = np.zeros(len(action)-1)
        executor_idx = self.idx_to_executor[idx]
        executor = self.executors[executor_idx]
        self.log.debug('isi:', executor.isi)
        predictor_list = executor.predictors
        for predictor_key in predictor_list:
            predictor = predictor_list[predictor_key]
            action[predictor.acc_type] += 1
        return action

    def apply_assignment(self, assignment):
        self.log.debug(f'Applying assignment: {assignment}')
        assignment = np.round(np.nan_to_num(
            assignment / np.sum(assignment, axis=0)) * self.predictors_max)
        new_assignment = np.zeros(assignment.shape)

        for idx in range(len(assignment)):
            cpu_pred, gpu_pred, vpu_pred, fpga_pred = assignment[idx]
            # look up executor idx and modify it
            isi_name = self.idx_to_executor[idx]
            executor = self.executors[isi_name]
            while cpu_pred > executor.num_predictor_types[AccType.CPU.value - 1]:
                # print('cpu_pred: ' + str(cpu_pred))
                # print(executor.num_predictor_types[AccType.CPU.value-1])
                executor.add_predictor(acc_type=AccType.CPU)
            while cpu_pred < executor.num_predictor_types[AccType.CPU.value - 1]:
                # print('cpu_pred: ' + str(cpu_pred))
                # print(executor.num_predictor_types[AccType.CPU.value-1])
                executor.remove_predictor_by_type(acc_type=AccType.CPU)
            new_assignment[idx][0] = executor.num_predictor_types[AccType.CPU.value - 1]

            while gpu_pred > executor.num_predictor_types[AccType.GPU.value - 1]:
                # print('gpu_pred: ' + str(gpu_pred))
                executor.add_predictor(acc_type=AccType.GPU)
            while gpu_pred < executor.num_predictor_types[AccType.GPU.value - 1]:
                # print('gpu_pred: ' + str(gpu_pred))
                executor.remove_predictor_by_type(acc_type=AccType.GPU)
            new_assignment[idx][1] = executor.num_predictor_types[AccType.GPU.value - 1]

            while vpu_pred > executor.num_predictor_types[AccType.VPU.value - 1]:
                executor.add_predictor(acc_type=AccType.VPU)
            while vpu_pred < executor.num_predictor_types[AccType.VPU.value - 1]:
                executor.remove_predictor_by_type(acc_type=AccType.VPU)
            new_assignment[idx][2] = executor.num_predictor_types[AccType.VPU.value - 1]

            while fpga_pred > executor.num_predictor_types[AccType.FPGA.value - 1]:
                executor.add_predictor(acc_type=AccType.FPGA)
            while fpga_pred < executor.num_predictor_types[AccType.FPGA.value - 1]:
                executor.remove_predictor_by_type(acc_type=AccType.FPGA)
            new_assignment[idx][3] = executor.num_predictor_types[AccType.FPGA.value - 1]

        self.log.debug(f'New applied assignment: {new_assignment}')

        if np.array_equal(new_assignment, assignment):
            self.log.debug('Was able to match given assignment exactly')
        else:
            self.log.debug(
                'Could not match given assignment exactly, had to scale')
        return

    def apply_assignment_vector(self, assignment):
        ''' Vector version of apply_assignment.
        '''
        self.log.debug(f'Assignment: {assignment}')
        idx = assignment[0]
        new_assignment = np.zeros(assignment.shape[0] - 1)

        for qos_level in range(self.n_qos_levels):
            self.log.debug('Assignment for QoS {}: {}'.format(
                qos_level, assignment[qos_level * 4 + 1:qos_level * 4 + 5]))
            cpu_pred, gpu_pred, vpu_pred, fpga_pred = assignment[qos_level *
                                                                 4 + 1:qos_level * 4 + 5]
            isi_name = self.idx_to_executor[idx]
            executor = self.executors[isi_name]

            while cpu_pred > executor.num_predictor_types[AccType.CPU.value - 1 + qos_level * 4] \
                    and self.available_predictors[0] > 0:
                # print('cpu_pred: {}'.format(cpu_pred))
                executor.add_predictor(
                    acc_type=AccType.CPU, qos_level=qos_level)
                self.available_predictors[0] -= 1
                # print('cpus: {}'.format(executor.num_predictor_types))
            while cpu_pred < executor.num_predictor_types[AccType.CPU.value - 1 + qos_level * 4]:
                # print('cpu_pred: {}'.format(cpu_pred))
                # print('< executor.num_predictor_types: {}'.format(executor.num_predictor_types[AccType.CPU.value-1]))
                executor.remove_predictor_by_type(
                    acc_type=AccType.CPU.value, qos_level=qos_level)
                self.available_predictors[0] += 1
                # print('cpus: {}'.format(executor.num_predictor_types))
            new_assignment[0 + qos_level *
                           4] = executor.num_predictor_types[AccType.CPU.value - 1 + qos_level * 4]

            while gpu_pred > executor.num_predictor_types[AccType.GPU.value - 1 + qos_level * 4] \
                    and self.available_predictors[1] > 0:
                # print('gpu_pred: {}'.format(gpu_pred))
                executor.add_predictor(
                    acc_type=AccType.GPU, qos_level=qos_level)
                self.available_predictors[1] -= 1
                # print('gpus: {}'.format(executor.num_predictor_types))
            while gpu_pred < executor.num_predictor_types[AccType.GPU.value - 1 + qos_level * 4]:
                # print('gpu_pred: {}'.format(gpu_pred))
                executor.remove_predictor_by_type(
                    acc_type=AccType.GPU.value, qos_level=qos_level)
                self.available_predictors[1] += 1
                # print('gpus: {}'.format(executor.num_predictor_types))
            new_assignment[1 + qos_level *
                           4] = executor.num_predictor_types[AccType.GPU.value - 1 + qos_level * 4]

            while vpu_pred > executor.num_predictor_types[AccType.VPU.value - 1 + qos_level * 4] \
                    and self.available_predictors[2] > 0:
                # print('vpu_pred: {}'.format(vpu_pred))
                executor.add_predictor(
                    acc_type=AccType.VPU, qos_level=qos_level)
                self.available_predictors[2] -= 1
            while vpu_pred < executor.num_predictor_types[AccType.VPU.value - 1 + qos_level * 4]:
                # print('vpu_pred: {}'.format(vpu_pred))
                executor.remove_predictor_by_type(
                    acc_type=AccType.VPU.value, qos_level=qos_level)
                self.available_predictors[2] += 1
            new_assignment[2 + qos_level *
                           4] = executor.num_predictor_types[AccType.VPU.value - 1 + qos_level * 4]

            while fpga_pred > executor.num_predictor_types[AccType.FPGA.value - 1 + qos_level * 4] \
                    and self.available_predictors[3] > 0:
                # print('fpga_pred: {}'.format(fpga_pred))
                executor.add_predictor(
                    acc_type=AccType.FPGA, qos_level=qos_level)
                self.available_predictors[3] -= 1
            while fpga_pred < executor.num_predictor_types[AccType.FPGA.value - 1 + qos_level * 4]:
                # print('fpga_pred: {}'.format(fpga_pred))
                executor.remove_predictor_by_type(
                    acc_type=AccType.FPGA.value, qos_level=qos_level)
                self.available_predictors[3] += 1
            new_assignment[3 + qos_level *
                           4] = executor.num_predictor_types[AccType.FPGA.value - 1 + qos_level * 4]

        self.log.debug('Old assignment for executor {}: {}'.format(
            assignment[0], assignment[1:]))
        self.log.debug('New applied assignment: {}'.format(new_assignment))
        self.log.debug('Remaining available predictors: {}'.format(
            self.available_predictors))

        return new_assignment

    def get_available_predictors(self):
        return self.available_predictors

    def evaluate_reward(self, K):
        '''
        RL-Cache version of reward: play the trace out K steps into the future,
        find the # of missed requests, and then roll back
        '''
        self.log.debug(
            '--- temp_simulator: Playing trace into the future to evaluate reward ---')

        if len(self.event_queue) == 0:
            self.log.warn('WARN: Ran out of requests while evaluating reward.')

        # TODO: turn back on when using RL
        return 0

        # only copying partial event queue
        partial_event_queue = copy.deepcopy(self.event_queue[:K * 2])
        temp_event_queue = self.event_queue
        self.event_queue = partial_event_queue

        # not deepcopying references to trace files
        temp_trace_files = self.trace_files
        self.trace_files = {}

        temp_simulator = copy.deepcopy(self)

        self.event_queue = temp_event_queue
        self.trace_files = temp_trace_files

        # TODO: only play forward for that particular ISI
        temp_simulator.simulate_requests(requests=K)
        self.log.debug('Failed requests: {}'.format(
            temp_simulator.failed_requests))
        self.log.debug('--- temp_simulator: Rolling back ---')
        reward = -temp_simulator.failed_requests
        return reward

    def process(self, event, clock):
        '''
        Process the given event according to its EventType.
        '''
        if event.type == EventType.START_REQUEST:
            self.process_start_event(event, clock)
        elif event.type == EventType.END_REQUEST:
            self.process_end_event(event, clock)
        elif event.type == EventType.SCHEDULING:
            self.process_scheduling_event(event, clock)
        elif event.type == EventType.FINISH_BATCH:
            self.process_finish_batch_event(event, clock)
        elif event.type == EventType.SLO_EXPIRING:
            self.process_slo_expiring_event(event, clock)

        return None

    def process_start_event(self, event, clock):
        ''' Process EventType.START_REQUEST
        '''
        self.log.debug('Starting event {}. (Time: {})'.format(event.desc, clock))
        isi = event.desc
        if isi not in self.executors:
            self.add_executor(isi, self.job_sched_algo, self.runtimes, self.model_variant_runtimes,
                                self.model_variant_loadtimes, max_acc_per_type=self.max_acc_per_type,
                                simulator=self)

        # call executor.process_request() on relevant executor
        executor = self.executors[isi]

        if not self.batching_enabled:
            self.process_start_event_without_batch(event, clock, executor)
        else:
            self.process_start_event_with_batch(event, clock, executor)
        
        # Bump overall request statistics
        self.bump_overall_request_stats(event)
        return

    def process_start_event_without_batch(self, event, clock, executor):
        ''' If we do not have batching enabled, we will immediately know
        whether an incoming request can be processed or not. This function
        handles the simulator's behavior for the no-batching scenario.
        '''
        end_time, qos_met, accuracy_seen = executor.process_request(
            event, clock, self.runtimes)

        # We report QoS failure whenever:
        # (i) request fails, (ii) request succeeds but QoS is not met

        if end_time is None:
            # Cannot process request, no end event generated, bump failure stats
            self.bump_failed_request_stats(event)
            self.log.debug('WARN: Request id {} for {} failed. (Time: {})'.format(
                event.id, event.desc, clock))
        else:
            # Request can be processed, generate an end event for it
            # TODO: look into dequeue
            self.generate_end_request_event(event, end_time, accuracy_seen, qos_met)
            
        return

    def process_start_event_with_batch(self, event, clock, executor):
        ''' If batching is enabled, the request goes through a different pipeline
        and we will not immediately know if request gets processed. So we simply
        add the request to an appropriate predictor.
        '''
        # TODO: bump failed request or qos unmet stats at appropriate points
        #       wthin the execution pipeline of batched requests 
        executor.enqueue_request(event, clock)
        return

    def generate_end_request_event(self, event, end_time, accuracy_seen, qos_met):
        ''' Generate an END_REQUEST event and insert it into the simulator's
        event queue.
        '''
        self.total_accuracy += accuracy_seen
        isi = event.desc
        self.insert_event(end_time, EventType.END_REQUEST,
                        event.desc, id=event.id, qos_level=event.qos_level,
                        event_counter=event.start_time)
        self.accuracy_logfile.write(str(accuracy_seen) + '\n')
        self.accuracy_per_model[isi].append(accuracy_seen)
        
        if not qos_met:
            self.bump_qos_unmet_stats(event)

        return

    def generate_finish_batch_event(self, finish_time, predictor, executor):
        ''' Generate a FINISH_BATCH event and insert it into the simulator's
        event queue
        '''
        self.insert_event(start_time=finish_time, type=EventType.FINISH_BATCH,
                        desc='finish_batch', predictor=predictor,
                        executor=executor)
        return
    
    def process_finish_batch_event(self, event, clock):
        ''' When a FINISH_BATCH event is encountered, trigger the appropriate
        callbacks
        '''
        executor = event.executor
        predictor = event.predictor

        # TODO: It is possible that predictor might have been removed by this point
        #       how to handle this case?

        predictor.finish_batch_callback(clock)
        return

    def generate_slo_expiring_event(self, time, request, predictor, executor, event_counter):
        ''' Generate an SLO_EXPIRING event and insert it into the simulator's
        event queue
        '''
        self.insert_event(time, EventType.SLO_EXPIRING, desc=request.desc,
                        id=request.id, deadline=request.deadline,
                        predictor=predictor, executor=executor,
                        event_counter=event_counter)
        return

    def process_slo_expiring_event(self, event, clock):
        ''' When an SLO_EXPIRING event is encountered, trigger the appropriate
        callbacks
        '''
        executor = event.executor
        predictor = event.predictor

        # TODO: It is possible that predictor might have been removed by this point
        #       how to handle this case?

        predictor.slo_expiring_callback(event, clock)
        return

    def process_end_event(self, event, clock):
        ''' Process EventType.END_REQUEST
        '''
        # self.log.debug('Event {} ended. (Time: {})'.format(
        #     event.desc, event.start_time))
        isi = event.desc
        executor = self.executors[isi]
        request_finished = executor.finish_request(event, clock)
        if request_finished:
            self.bump_successful_request_stats(event)
            finish_time = clock - event.event_counter
            self.latency_logfile.write(f'{isi},{event.event_counter},{clock},{finish_time}\n')
        else:
            self.bump_failed_request_stats(event)
        return

    def process_scheduling_event(self, event, clock):
        ''' Process EventType.SCHEDULING
        '''
        sched_decision = self.invoke_scheduling_agent()
        if sched_decision is not None:
            self.log.debug(sched_decision)
        else:
            self.log.error('ERROR: The scheduling agent returned an exception. (Time: {})'.format(
                event.start_time))
        self.insert_event(clock + self.sched_interval,
                            EventType.SCHEDULING, event.desc)
        return

    def bump_failed_request_stats(self, event):
        ''' Bump stats for a failed request
        '''
        self.log.debug(f'Failed request: {event.desc}')
        # time.sleep(1)
        isi = event.desc
        self.failed_requests += 1
        self.total_failed += 1
        
        self.failed_requests_arr[self.isi_to_idx[isi]] += 1
        self.failed_per_model[isi] += 1
        self.slo_timeouts['timeouts'] += 1
        self.bump_qos_unmet_stats(event)
        return

    def bump_qos_unmet_stats(self, event):
        ''' Bump stats for unmet QoS on request
        '''
        isi = event.desc
        self.qos_stats[self.isi_to_idx[isi], event.qos_level * 2 + 1] += 1
        return
    
    def bump_successful_request_stats(self, event):
        ''' Bump stats for a successful request
        '''
        isi = event.desc
        self.completed_requests += 1
        self.successful_requests += 1
        self.successful_requests_arr[self.isi_to_idx[isi]] += 1
        self.successful_per_model[isi] += 1
        self.slo_timeouts['succeeded'] += 1
        self.total_successful += 1
        return

    def bump_overall_request_stats(self, event):
        ''' Bump overall request stats, i.e., total requests per ISI, requests
        per model, and QoS stats
        '''
        isi = event.desc
        self.total_requests_arr[self.isi_to_idx[isi]] += 1
        self.requests_per_model[isi] += 1
        self.qos_stats[self.isi_to_idx[isi], event.qos_level * 2] += 1
        # self.test_stat += 1
        return

    def add_executor(self, isi, job_sched_algo, runtimes=None, model_variant_runtimes=None, 
                        model_variant_loadtimes=None, max_acc_per_type=0, simulator=None):
        executor = Executor(isi, job_sched_algo, self.logging_level, self.n_qos_levels,
                            runtimes, model_variant_runtimes, model_variant_loadtimes,
                            max_acc_per_type=max_acc_per_type, simulator=simulator)
        self.executors[executor.isi] = executor
        return executor.id

    def insert_event(self, start_time, type, desc, runtime=None, id='', deadline=1000,
                    qos_level=0, accuracy=None, predictor=None, executor=None,
                    event_counter=0):
        if type == EventType.END_REQUEST:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                            qos_level=qos_level, event_counter=event_counter)
        elif type == EventType.FINISH_BATCH:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                            qos_level=qos_level, accuracy=accuracy, predictor=predictor,
                            executor=executor)
        elif type == EventType.SLO_EXPIRING:
            event = Event(start_time, type, desc, runtime, id=id, deadline=deadline,
                            qos_level=qos_level, accuracy=accuracy, predictor=predictor,
                            executor=executor, event_counter=event_counter)
        elif type == EventType.START_REQUEST:
            event = Event(start_time, type, desc, runtime, deadline=deadline,
                            qos_level=qos_level, accuracy=accuracy)
        else:
            event = Event(start_time, type, desc, runtime, deadline=deadline,
                            qos_level=qos_level, accuracy=accuracy)

        # using binary search to reduce search time complexity
        idx = self.binary_find_index(self.event_queue, start_time)
        self.event_queue.insert(idx, event)
        inserted = True

        # if type == EventType.START_REQUEST:
        #     print(f'idx: {idx}, length of event queue: {len(self.event_queue)}')
        #     time.sleep(0.5)
        # print(f'event_queue: {list(map(lambda x: (x.start_time, x.type), self.event_queue))}')
        # print(f'length of event queue: {len(self.event_queue)}')
        # time.sleep(0.0001)

        return inserted

    def linear_find_index(self, arr, number):
        size = len(arr)
        # Traverse the array
        for i in range(size):
            # If number is found
            if arr[i].start_time == number:
                return i
            # If arr[i] exceeds number
            elif arr[i].start_time > number:
                return i
        # If all array elements are smaller
        return size

    def binary_find_index(self, arr, number):
        # Lower and upper bounds
        start = 0
        end = len(arr) - 1

        # Traverse the search space
        while start <= end:
            mid = (start + end) // 2
            if arr[mid].start_time == number:
                return mid
            elif arr[mid].start_time < number:
                start = mid + 1
            else:
                end = mid - 1
        # Return the insert position
        return end + 1

    def invoke_scheduling_agent(self):
        if self.mode == 'debugging':
            # just for readability purposes
            time.sleep(2)

        self.log.info('Invoking scheduling event on scheduling_agent')
        print('Invoking scheduling event on scheduling_agent')

        request_url = self.sched_agent_uri + '/scheduling_event'
        request = None
        try:
            request = requests.get(request_url)
            self.log.debug(request.content)
            # TODO: perhaps parse request content before passing back?
            return request.content
        except requests.exceptions.ConnectionError as connException:
            self.log.error(connException)
            return None

    def print(self):
        for i in range(len(self.event_queue)):
            event = self.event_queue[i]
            self.log.debug('Time: {}, event {} of type {}'.format(
                event.start_time, event.desc, event.type))


if __name__ == '__main__':
    simulator = Simulator(trace_path='/home/soahmad/blis/evaluation/scheduler/traces/test/',
                          mode='debugging')

    # simulator.insert_event(2, EventType.START_REQUEST, 'resnet-test', runtime=10, deadline=50)
    # simulator.insert_event(3, EventType.START_REQUEST, 'resnet-test', runtime=10, deadline=50)
    # simulator.insert_event(3, EventType.START_REQUEST, 'resnet-test', runtime=10, deadline=50)
    # simulator.insert_event(3, EventType.START_REQUEST, 'resnet-test', runtime=10, deadline=50)
    # simulator.insert_event(3, EventType.START_REQUEST, 'resnet-test', runtime=10, deadline=50)
    # simulator.insert_event(3, EventType.START_REQUEST, 'resnet-test', runtime=10, deadline=50)
    # simulator.insert_event(3, EventType.START_REQUEST, 'ssd-test', runtime=7, deadline=50)

    # we should never insert EventType.END_REQUEST events, they will be
    # automatically created by simulator
    # simulator.insert_event(8, EventType.END_REQUEST, 'checking_end')

    # simulator.print()

    simulator.simulate_until(until=8)
    print('Reward: {}'.format(simulator.evaluate_reward(10)))
    simulator.failed_requests = 0
    # simulator.apply_assignment(np.random.randint(0, 11, size=(2,4)))
    simulator.apply_assignment_vector(np.random.randint(0, 2, size=(5)))
    print('Reward: {}'.format(simulator.evaluate_reward(10)))
    print('__main__ -- Applied assignment')
    simulator.simulate_until(until=-1)
