import copy
import logging
import numpy as np
import random
import requests
import os
import time
from hashlib import new
from pyexpat import model
from core.event import Event, EventType
from core.executor import Executor, AccType


class Simulator:
    def __init__(self, job_sched_algo, trace_path=None, mode='training', 
                 max_acc_per_type=0, predictors_max=[10, 10, 10, 10], n_qos_levels=1, random_runtimes=False,
                 fixed_seed=0):
        self.clock = 0
        self.event_queue = []
        self.executors = {}
        self.idx_to_executor = {}
        self.isi_to_idx = {}
        self.failed_requests = 0
        self.failed_requests_arr = []
        self.total_requests_arr = []
        self.mode = mode
        self.completed_requests = 0

        self.max_acc_per_type = max_acc_per_type

        if fixed_seed != 0:
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)

        self.job_sched_algo = job_sched_algo
        self.n_qos_levels = n_qos_levels

        self.store_file_pointers = True
        self.requests_added = 0

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

        self.accuracy_per_model = {}

        self.trace_files = {}

        self.accuracy_logfile = open('logs/log_ilp_accuracy.txt', mode='w')

        self.predictors_max = predictors_max
        # self.available_predictors = np.tile(copy.deepcopy(predictors_max), self.n_qos_levels)
        self.available_predictors = copy.deepcopy(predictors_max)

        self.batching_enabled = False

        logging.basicConfig(level=logging.INFO)

        idx = 0
        if not trace_path is None:
            logging.info('Reading trace files from: ' + trace_path)
            variant_list_path = os.path.join(trace_path, '..', 'model_variants')
            trace_files = sorted(os.listdir(trace_path))

            for file in trace_files:
                filename = file.split('/')[-1]
                if len(filename.split('.')) == 1:
                    # it is a directory
                    continue
                isi_name, extension = filename.split('.')
                if not extension == 'txt':
                    continue
                logging.info('Filename: ' + file)

                variant_list_filename = os.path.join(variant_list_path, isi_name)
                logging.info('Variant list filename:' + variant_list_filename)

                model_variant_list = self.read_variants_from_file(variant_list_filename)
                self.set_model_variants(isi_name, model_variant_list)

                # This random initialization has been replaced by reading from file in read_variants_from_file()
                # self.set_model_variant_accuracies(isi_name, filename='')

                self.initialize_runtimes(
                    isi_name, random_runtimes=random_runtimes)
                self.initialize_model_variant_loadtimes(isi_name)
                self.initialize_model_variant_runtimes(isi_name, random_runtimes=random_runtimes)

                self.add_executor(isi_name, self.job_sched_algo, runtimes={},
                                    model_variant_runtimes={}, model_variant_loadtimes={},
                                    max_acc_per_type=self.max_acc_per_type, simulator=self)
                self.idx_to_executor[idx] = isi_name
                self.isi_to_idx[isi_name] = idx
                self.failed_requests_arr.append(0)
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
                    logging.debug(
                        'Time to add trace file: {} seconds'.format(end - start))

        self.runtimes = {1: self.cpu_runtimes, 2: self.gpu_runtimes,
                         3: self.vpu_runtimes, 4: self.fpga_runtimes}

        self.loadtimes = {1: self.cpu_loadtimes, 2: self.gpu_loadtimes,
                          3: self.vpu_loadtimes, 4: self.fpga_loadtimes}

        self.model_variant_runtimes = {1: self.cpu_variant_runtimes, 2: self.gpu_variant_runtimes,
                                        3: self.vpu_variant_runtimes, 4: self.fpga_variant_runtimes}

        self.set_executor_runtimes()
        self.set_executor_loadtimes()
        logging.info('Model variant accuracies: {}'.format(self.model_variant_accuracies))

        self.set_executor_model_variants()

        self.set_executor_variant_accuracies()
        self.set_executor_variant_runtimes()
        self.set_executor_variant_loadtimes()

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
        if os.path.exists(filename):
            with open(filename, mode='r') as rf:
                for line in rf.readlines():
                    model_variant = line.split(',')[0]
                    accuracy = float(line.rstrip('\n').split(',')[1])

                    model_variants.append(model_variant)
                    self.model_variant_accuracies[(isi_name, model_variant)] = accuracy
        else:
            logging.error('read_variants_from_file: no path {} found!'.format(filename))
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
        return False

    def set_model_variants(self, isi_name, model_variant_list):
        self.model_variants[isi_name] = model_variant_list

        self.requests_per_model[isi_name] = 0
        self.failed_per_model[isi_name] = 0

        self.accuracy_per_model[isi_name] = []

    def initialize_runtimes(self, isi_name, random_runtimes=False):
        for qos_level in range(self.n_qos_levels):
            if random_runtimes:
                self.cpu_runtimes[isi_name,
                                  qos_level] = random.randint(20, 100)
                self.gpu_runtimes[isi_name,
                                  qos_level] = random.randint(20, 100)
                self.vpu_runtimes[isi_name,
                                  qos_level] = random.randint(20, 100)
                self.fpga_runtimes[isi_name,
                                   qos_level] = random.randint(20, 100)
            else:
                self.cpu_runtimes[isi_name, qos_level] = 50
                self.gpu_runtimes[isi_name, qos_level] = 25
                self.vpu_runtimes[isi_name, qos_level] = 35
                self.fpga_runtimes[isi_name, qos_level] = 30
                # self.cpu_runtimes[isi_name, qos_level] = random.randint(25, 100)
                # self.gpu_runtimes[isi_name, qos_level] = random.randint(25, 100)
                # self.vpu_runtimes[isi_name, qos_level] = random.randint(25, 100)
                # self.fpga_runtimes[isi_name, qos_level] = random.randint(25, 100)
            # print(self.cpu_runtimes)
            # print(self.gpu_runtimes)
            # print(self.vpu_runtimes)
            # print(self.fpga_runtimes)
        return

    def get_thput_accuracy_per_model(self):
        requests = []
        failed = []
        accuracy = []
        for model in self.requests_per_model:
            requests.append(self.requests_per_model[model])
            self.requests_per_model[model] = 0
            
            failed.append(self.failed_per_model[model])
            self.failed_per_model[model] = 0

            total_accuracy = sum(self.accuracy_per_model[model])
            accuracy.append(total_accuracy)
            self.accuracy_per_model[model] = []

        return requests, failed, accuracy

    def initialize_model_variant_loadtimes(self, isi_name):
        model_variant_list = self.model_variants[isi_name]
        for model_variant in model_variant_list:
            self.model_variant_loadtimes[(isi_name, model_variant)] = 0
    
    def initialize_model_variant_runtimes(self, isi_name, random_runtimes=False):
        model_variant_list = self.model_variants[isi_name]
        for model_variant in model_variant_list:
            if random_runtimes:
                self.cpu_variant_runtimes[(isi_name, model_variant)] = random.randint(20, 100)
                self.gpu_variant_runtimes[(isi_name, model_variant)] = random.randint(20, 100)
                self.vpu_variant_runtimes[(isi_name, model_variant)] = random.randint(20, 100)
                self.fpga_variant_runtimes[(isi_name, model_variant)] = random.randint(20, 100)
            else:
                self.cpu_variant_runtimes[(isi_name, model_variant)] = 50
                self.gpu_variant_runtimes[(isi_name, model_variant)] = 25
                self.vpu_variant_runtimes[(isi_name, model_variant)] = 35
                self.fpga_variant_runtimes[(isi_name, model_variant)] = 30
    
    def set_model_variant_accuracies(self, isi_name, filename=''):
        if filename == '':
            # We set accuracies randomly
            for model_variant in self.model_variants[isi_name]:
                self.model_variant_accuracies[(isi_name, model_variant)] = random.uniform(50.0, 99.99)
        else:
            # Need to add support for reading accuracies from a file
            logging.error('Reading accuracies from file not implemented!')
            time.sleep(10)

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

    def trigger_infaas_downscaling(self):
        for key in self.executors:
            executor = self.executors[key]
            executor.trigger_infaas_downscaling()

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

            # in case we are loading file pointers instead of entire file
            if self.store_file_pointers:
                finished = self.refill_event_queue()
                if finished:
                    break
        return

    def refill_event_queue(self):
        if len(self.trace_files) == 0:
            # we are in a deepcopied simulator instance
            return False
        if len(self.event_queue) < 10000:
            finished = True
            for isi_name in self.trace_files:
                readfile = self.trace_files[isi_name]
                # print('Adding requests from {}'.format(readfile))
                file_finished = self.add_requests_from_trace_pointer(isi_name, readfile,
                                                                     read_until=self.clock + 10000)
                if file_finished:
                    logging.debug('Trace file {} finished'.format(isi_name))
                    # time.sleep(5000)
                finished = finished and file_finished
            return finished
        return False

    def is_done(self):
        '''
        Returns true if the request trace has finished, false otherwise.
        '''
        if len(self.event_queue) == 0:
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
        self.qos_stats = np.zeros((len(self.executors), self.n_qos_levels * 2))

    def get_total_requests_arr(self):
        return self.total_requests_arr

    def get_qos_stats(self):
        return self.qos_stats

    def get_failed_requests_arr(self):
        return self.failed_requests_arr

    def get_failed_requests(self):
        return self.failed_requests

    # def reset_failed_requests(self):
    #     self.failed_requests = 0
        # self.qos_stats = np.zeros((len(self.executors), self.n_qos_levels))

    def print_assignment(self):
        logging.debug('Printing assignment in simulator...')
        for isi in self.executors:
            assignment = self.executors[isi].num_predictor_types
            logging.debug(
                'Executor {} has assignment: {}'.format(isi, assignment))
        return

    def apply_predictor_dict(self, required_predictors={}, canary_dict={}):
        """ Takes in a dictionary 'required_predictors' with
        key: (model_variant, accelerator_type)
        value: number of predictors
        A dictionary 'canary_dict' with
        key: (model_variant, isi)
        value: canary routing percentage
        Applies the assignment to current system
        """
        if len(required_predictors) == 0:
            print('No required predictors passed')
            time.sleep(5)
            return

        ilp_to_acc_type = {}
        ilp_to_acc_type['CPU'] = 1
        ilp_to_acc_type['GPU_PASCAL'] = 2
        ilp_to_acc_type['VPU'] = 3
        ilp_to_acc_type['GPU_AMPERE'] = 4

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

        existing_count = {}
        for key in existing_predictors:
            (model_variant, acc_type) = key
            existing_count[(model_variant, AccType(acc_type))] = len(existing_predictors[key])
        # print('existing predictor count:', sorted(existing_count))
        # print('required predictors:', sorted(required_predictors))
        # time.sleep(3)

        # Go through each key:
        for tuple_key in required_predictors:
            # Remove predictors if needed
            (model_variant, accelerator_type) = tuple_key
            acc_type = ilp_to_acc_type[accelerator_type]

            if not((model_variant, acc_type) in existing_predictors):
                continue
            existing = len(existing_predictors[(model_variant, acc_type)])
            required = required_predictors[tuple_key]

            while existing > required:
                predictor = existing_predictors[(model_variant, acc_type)].pop()
                id = predictor.id
                executor = predictor.executor
                if (executor.remove_predictor_by_id(id)):
                    print('removed predictor {} of type {} for model variant {}'.format(id,
                                predictor.acc_type, model_variant))
                    self.available_predictors[acc_type-1] += 1
                else:
                    print('no predictor found with id:', id)
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
                    print('added predictor of type {} for model variant {}'.format(acc_type, model_variant))
                    added += 1
                if (self.available_predictors[acc_type-1]) < 0:
                    print('Error! Available predictors {} for type {}'.format(self.available_predictors[acc_type-1], acc_type))
                    time.sleep(5)
                required -= 1
        print('added predictors:', added)
        # time.sleep(1)
        
        return

    def null_action(self, action, idx):
        action[0] = idx
        action[1:] = np.zeros(len(action)-1)
        executor_idx = self.idx_to_executor[idx]
        executor = self.executors[executor_idx]
        print('isi:', executor.isi)
        predictor_list = executor.predictors
        for predictor_key in predictor_list:
            predictor = predictor_list[predictor_key]
            action[predictor.acc_type] += 1
        return action

    def apply_assignment(self, assignment):
        logging.debug('Applying assignment: {}'.format(assignment))
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

        logging.debug('New applied assignment: {}'.format(new_assignment))

        if np.array_equal(new_assignment, assignment):
            logging.debug('Was able to match given assignment exactly')
        else:
            logging.debug(
                'Could not match given assignment exactly, had to scale')
        return

    def apply_assignment_vector(self, assignment):
        ''' Vector version of apply_assignment.
        '''
        logging.debug('Assignment: {}'.format(assignment))
        idx = assignment[0]
        new_assignment = np.zeros(assignment.shape[0] - 1)

        for qos_level in range(self.n_qos_levels):
            logging.debug('Assignment for QoS {}: {}'.format(
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

        logging.debug('Old assignment for executor {}: {}'.format(
            assignment[0], assignment[1:]))
        logging.debug('New applied assignment: {}'.format(new_assignment))
        logging.debug('Remaining available predictors: {}'.format(
            self.available_predictors))

        return new_assignment

    def get_available_predictors(self):
        return self.available_predictors

    def evaluate_reward(self, K):
        '''
        RL-Cache version of reward: play the trace out K steps into the future,
        find the # of missed requests, and then roll back
        '''
        logging.debug(
            '--- temp_simulator: Playing trace into the future to evaluate reward ---')

        if len(self.event_queue) == 0:
            logging.warn('WARN: Ran out of requests while evaluating reward.')

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
        logging.debug('Failed requests: {}'.format(
            temp_simulator.failed_requests))
        logging.debug('--- temp_simulator: Rolling back ---')
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

        return None

    def process_start_event(self, event, clock):
        ''' Process EventType.START_REQUEST
        '''
        logging.debug('Starting event {}. (Time: {})'.format(event.desc, clock))
        isi = event.desc
        if isi not in self.executors:
            self.add_executor(isi, self.job_sched_algo, self.runtimes, self.model_variant_runtimes,
                                self.model_variant_loadtimes, max_acc_per_type=self.max_acc_per_type,
                                simulator=self)

        # call executor.process_request() on relevant executor
        executor = self.executors[isi]

        if not self.batching_enabled:
            end_time, qos_met, accuracy_seen = executor.process_request(
                event, clock, self.runtimes)
        else:
            # TODO: fix the return types for this function
            end_time, qos_met, accuracy_seen = executor.enqueue_request(
                event, clock)

        # Note: we report QoS failure whenever
        # (i) request fails, (ii) request succeeds but QoS is not met

        if end_time is None:
            # Cannot process request, no end event generated, bump failure stats
            self.bump_failed_request_stats(event)
            logging.debug('WARN: Request id {} for {} failed. (Time: {})'.format(
                event.id, event.desc, clock))
        else:
            # Request can be processed, generate an end event for it
            # TODO: look into dequeue
            self.insert_event(end_time, EventType.END_REQUEST,
                                event.desc, id=event.id, qos_level=event.qos_level)
            self.accuracy_logfile.write(str(accuracy_seen) + '\n')
            self.accuracy_per_model[isi].append(accuracy_seen)
            
            if not qos_met:
                self.bump_qos_unmet_stats(event)
        
        # Bump overall request statistics
        self.bump_overall_request_stats(event)
        return

    def process_end_event(self, event, clock):
        ''' Process EventType.END_REQUEST
        '''
        logging.debug('Event {} ended. (Time: {})'.format(
            event.desc, event.start_time))
        isi = event.desc
        executor = self.executors[isi]
        executor.finish_request(event, clock)
        self.completed_requests += 1
        return

    def process_scheduling_event(self, event, clock):
        ''' Process EventType.SCHEDULING
        '''
        sched_decision = self.invoke_scheduling_agent()
        if sched_decision is not None:
            logging.debug(sched_decision)
        else:
            logging.error('ERROR: The scheduling agent returned an exception. (Time: {})'.format(
                event.start_time))
        self.insert_event(clock + self.sched_interval,
                            EventType.SCHEDULING, event.desc)
        return

    def bump_failed_request_stats(self, event):
        ''' Bump stats for a failed request
        '''
        isi = event.desc
        self.failed_requests += 1
        self.failed_requests_arr[self.isi_to_idx[isi]] += 1
        self.failed_per_model[isi] += 1
        self.bump_qos_unmet_stats(event)
        return

    def bump_qos_unmet_stats(self, event):
        ''' Bump stats for unmet QoS on request
        '''
        isi = event.desc
        self.qos_stats[self.isi_to_idx[isi], event.qos_level * 2 + 1] += 1

    def bump_overall_request_stats(self, event):
        ''' Bump overall request stats, i.e., total requests per ISI, requests
        per model, and QoS stats
        '''
        isi = event.desc
        self.total_requests_arr[self.isi_to_idx[isi]] += 1
        self.requests_per_model[isi] += 1
        self.qos_stats[self.isi_to_idx[isi], event.qos_level * 2] += 1
        return

    def add_executor(self, isi, job_sched_algo, runtimes=None, model_variant_runtimes=None, 
                        model_variant_loadtimes=None, max_acc_per_type=0, simulator=None):
        executor = Executor(isi, job_sched_algo, self.n_qos_levels, runtimes,
                                model_variant_runtimes, model_variant_loadtimes,
                                max_acc_per_type=max_acc_per_type, simulator=simulator)
        self.executors[executor.isi] = executor
        return executor.id

    def insert_event(self, time, type, desc, runtime=None, id='', deadline=1000, qos_level=0, accuracy=None):
        if type == EventType.END_REQUEST:
            event = Event(time, type, desc, runtime, id=id, deadline=deadline,
                            qos_level=qos_level)
        else:
            event = Event(time, type, desc, runtime, deadline=deadline,
                            qos_level=qos_level, accuracy=accuracy)

        # using binary search to reduce search time complexity
        idx = self.binary_find_index(self.event_queue, time)
        self.event_queue.insert(idx, event)
        inserted = True

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

        logging.info('Invoking scheduling event on scheduling_agent')
        print('Invoking scheduling event on scheduling_agent')

        request_url = self.sched_agent_uri + '/scheduling_event'
        request = None
        try:
            request = requests.get(request_url)
            print(request.content)
            # TODO: perhaps parse request content before passing back?
            return request.content
        except requests.exceptions.ConnectionError as connException:
            print(connException)
            return None

    def print(self):
        for i in range(len(self.event_queue)):
            event = self.event_queue[i]
            print('Time: {}, event {} of type {}'.format(
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
