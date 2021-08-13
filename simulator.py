import copy
import glob
from hashlib import new
import logging
import random
import requests
import os
import time
import uuid
import numpy as np
from enum import Enum
from executor import Executor, AccType


class Simulator:
    def __init__(self, trace_path=None, mode='training', predictors_max=[10,10,10,10], n_qos_levels=1):
        self.clock = 0
        self.event_queue = []
        self.executors = {}
        self.idx_to_executor = {}
        self.isi_to_idx = {}
        self.failed_requests = 0
        self.failed_requests_arr = []
        self.total_requests_arr = []
        self.mode = mode

        self.n_qos_levels = n_qos_levels

        self.store_file_pointers = True
        self.requests_added = 0

        self.cpu_runtimes = {}
        self.gpu_runtimes = {}
        self.vpu_runtimes = {}
        self.fpga_runtimes = {}

        self.trace_files = {}

        self.predictors_max = predictors_max
        # self.available_predictors = np.tile(copy.deepcopy(predictors_max), self.n_qos_levels)
        self.available_predictors = copy.deepcopy(predictors_max)

        logging.basicConfig(level=logging.INFO)

        idx = 0
        if not trace_path is None:
            trace_files = os.listdir(trace_path)
            for file in trace_files:
                filename = file.split('/')[-1]
                if len(filename.split('.')) == 1:
                    # it is a directory
                    continue
                isi_name, extension = filename.split('.')
                if not extension == 'txt':
                    continue
                logging.info('Filename: ' + file)
                self.add_executor(isi_name)
                self.idx_to_executor[idx] = isi_name
                self.isi_to_idx[isi_name] = idx
                self.failed_requests_arr.append(0)
                self.total_requests_arr.append(0)
                idx += 1
            
                self.initialize_random_runtimes(isi_name)
                
                if self.store_file_pointers:
                    readfile = open(os.path.join(trace_path, file), mode='r')
                    self.trace_files[isi_name] = readfile
                    self.add_requests_from_trace_pointer(isi_name, readfile, read_until=10000)
                else:
                    start = time.time()
                    self.add_requests_from_trace(isi_name, os.path.join(trace_path, file))
                    end = time.time()
                    logging.debug('Time to add trace file: {} seconds'.format(end-start))

        self.runtimes = {1: self.cpu_runtimes, 2: self.gpu_runtimes,
                         3: self.vpu_runtimes, 4: self.fpga_runtimes}

        self.qos_stats = np.zeros((len(self.executors), n_qos_levels*2))
        
        if not (mode == 'training' or mode == 'debugging'):
            # if the RL agent is being trained, we don't need to invoke scheduling microservice
            self.insert_event(0, EventType.SCHEDULING, 'invoke scheduling agent')

        # how frequently the scheduling agent is invoked (in milliseconds)
        self.sched_interval = 200

        self.sched_agent_uri = 'http://127.0.0.1:8000'

    
    def reset(self):
        return


    def add_requests_from_trace(self, isi_name, file):
        ''' Completely read trace file into memory.
        '''
        with open(file, mode='r') as readfile:
            for line in readfile:
                request_description = line.rstrip('\n').split(',')
                start_time = int(request_description[0])
                # if qos_level is defined, use that. otherwise use qos_level 0 by default
                if len(request_description) == 2:
                    qos_level = int(request_description[1])
                else:
                    qos_level = 0
                # also, if n_qos_levels is 1, ignore qos info from trace
                if self.n_qos_levels == 1:
                    qos_level = 0
                self.insert_event(start_time, EventType.START_REQUEST, isi_name, runtime=None,
                                  deadline=500, qos_level=qos_level)
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
            request_description = line.rstrip('\n').split(',')
            start_time = int(request_description[0])
            # if qos_level is defined, use that. otherwise use qos_level 0 by default
            if len(request_description) == 2:
                qos_level = int(request_description[1])
            else:
                qos_level = 0
            # also, if n_qos_levels is 1, ignore qos info from trace
            if self.n_qos_levels == 1:
                qos_level = 0
            self.insert_event(start_time, EventType.START_REQUEST, isi_name, runtime=None,
                                deadline=500, qos_level=qos_level)
            self.requests_added += 1
            if start_time >= read_until:
                break
            # print(self.requests_added)
        return False

    
    def initialize_random_runtimes(self, isi_name):
        for qos_level in range(self.n_qos_levels):
            self.cpu_runtimes[isi_name, qos_level] = random.randint(100, 350)
            self.gpu_runtimes[isi_name, qos_level] = random.randint(100, 350)
            self.vpu_runtimes[isi_name, qos_level] = random.randint(100, 350)
            self.fpga_runtimes[isi_name, qos_level] = random.randint(100, 350)
        print
        return

    
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
                                                read_until=self.clock+10000)
                if file_finished:
                    logging.info('Trace file {} finished'.format(isi_name))
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
        self.qos_stats = np.zeros((len(self.executors), self.n_qos_levels*2))

    
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
            logging.debug('Executor {} has assignment: {}'.format(isi, assignment))
        return

    
    def apply_assignment(self, assignment):
        logging.debug('Applying assignment: {}'.format(assignment))
        assignment = np.round(np.nan_to_num(assignment / np.sum(assignment, axis=0)) * self.predictors_max)
        new_assignment = np.zeros(assignment.shape)

        for idx in range(len(assignment)):
            cpu_pred, gpu_pred, vpu_pred, fpga_pred = assignment[idx]
            # look up executor idx and modify it
            isi_name = self.idx_to_executor[idx]
            executor = self.executors[isi_name]
            while cpu_pred > executor.num_predictor_types[AccType.CPU.value-1]:
                # print('cpu_pred: ' + str(cpu_pred))
                # print(executor.num_predictor_types[AccType.CPU.value-1])
                executor.add_predictor(acc_type=AccType.CPU)
            while cpu_pred < executor.num_predictor_types[AccType.CPU.value-1]:
                # print('cpu_pred: ' + str(cpu_pred))
                # print(executor.num_predictor_types[AccType.CPU.value-1])
                executor.remove_predictor_by_type(acc_type=AccType.CPU)
            new_assignment[idx][0] = executor.num_predictor_types[AccType.CPU.value-1]
                
            while gpu_pred > executor.num_predictor_types[AccType.GPU.value-1]:
                # print('gpu_pred: ' + str(gpu_pred))
                executor.add_predictor(acc_type=AccType.GPU)
            while gpu_pred < executor.num_predictor_types[AccType.GPU.value-1]:
                # print('gpu_pred: ' + str(gpu_pred))
                executor.remove_predictor_by_type(acc_type=AccType.GPU)
            new_assignment[idx][1] = executor.num_predictor_types[AccType.GPU.value-1]
                
            while vpu_pred > executor.num_predictor_types[AccType.VPU.value-1]:
                executor.add_predictor(acc_type=AccType.VPU)
            while vpu_pred < executor.num_predictor_types[AccType.VPU.value-1]:
                executor.remove_predictor_by_type(acc_type=AccType.VPU)
            new_assignment[idx][2] = executor.num_predictor_types[AccType.VPU.value-1]
                
            while fpga_pred > executor.num_predictor_types[AccType.FPGA.value-1]:
                executor.add_predictor(acc_type=AccType.FPGA)
            while fpga_pred < executor.num_predictor_types[AccType.FPGA.value-1]:
                executor.remove_predictor_by_type(acc_type=AccType.FPGA)
            new_assignment[idx][3] = executor.num_predictor_types[AccType.FPGA.value-1]
        
        logging.debug('New applied assignment: {}'.format(new_assignment))

        if np.array_equal(new_assignment, assignment):
            logging.debug('Was able to match given assignment exactly')
        else:
            logging.debug('Could not match given assignment exactly, had to scale')
        return

    
    def apply_assignment_vector(self, assignment):
        ''' Vector version of apply_assignment.
        '''
        logging.debug('Assignment: {}'.format(assignment))
        idx = assignment[0]
        new_assignment = np.zeros(assignment.shape[0]-1)

        for qos_level in range(self.n_qos_levels):
            logging.debug('Assignment for QoS {}: {}'.format(qos_level, assignment[qos_level*4+1:qos_level*4+5]))
            cpu_pred, gpu_pred, vpu_pred, fpga_pred = assignment[qos_level*4+1:qos_level*4+5]
            isi_name = self.idx_to_executor[idx]
            executor = self.executors[isi_name]

            while cpu_pred > executor.num_predictor_types[AccType.CPU.value-1 + qos_level*4] \
                and self.available_predictors[0] > 0:
                # print('cpu_pred: {}'.format(cpu_pred))
                executor.add_predictor(acc_type=AccType.CPU, qos_level=qos_level)
                self.available_predictors[0] -= 1
                # print('cpus: {}'.format(executor.num_predictor_types))
            while cpu_pred < executor.num_predictor_types[AccType.CPU.value-1 + qos_level*4]:
                # print('cpu_pred: {}'.format(cpu_pred))
                # print('< executor.num_predictor_types: {}'.format(executor.num_predictor_types[AccType.CPU.value-1]))
                executor.remove_predictor_by_type(acc_type=AccType.CPU.value, qos_level=qos_level)
                self.available_predictors[0] += 1
                # print('cpus: {}'.format(executor.num_predictor_types))
            new_assignment[0 + qos_level*4] = executor.num_predictor_types[AccType.CPU.value-1 + qos_level*4]
                
            while gpu_pred > executor.num_predictor_types[AccType.GPU.value-1 + qos_level*4] \
                and self.available_predictors[1] > 0:
                # print('gpu_pred: {}'.format(gpu_pred))
                executor.add_predictor(acc_type=AccType.GPU, qos_level=qos_level)
                self.available_predictors[1] -= 1
                # print('gpus: {}'.format(executor.num_predictor_types))
            while gpu_pred < executor.num_predictor_types[AccType.GPU.value-1 + qos_level*4]:
                # print('gpu_pred: {}'.format(gpu_pred))
                executor.remove_predictor_by_type(acc_type=AccType.GPU.value, qos_level=qos_level)
                self.available_predictors[1] += 1
                # print('gpus: {}'.format(executor.num_predictor_types))
            new_assignment[1 + qos_level*4] = executor.num_predictor_types[AccType.GPU.value-1 + qos_level*4]
                
            while vpu_pred > executor.num_predictor_types[AccType.VPU.value-1 + qos_level*4] \
                and self.available_predictors[2] > 0:
                # print('vpu_pred: {}'.format(vpu_pred))
                executor.add_predictor(acc_type=AccType.VPU, qos_level=qos_level)
                self.available_predictors[2] -= 1
            while vpu_pred < executor.num_predictor_types[AccType.VPU.value-1 + qos_level*4]:
                # print('vpu_pred: {}'.format(vpu_pred))
                executor.remove_predictor_by_type(acc_type=AccType.VPU.value, qos_level=qos_level)
                self.available_predictors[2] += 1
            new_assignment[2 + qos_level*4] = executor.num_predictor_types[AccType.VPU.value-1 + qos_level*4]
                
            while fpga_pred > executor.num_predictor_types[AccType.FPGA.value-1 + qos_level*4] \
                and self.available_predictors[3] > 0:
                # print('fpga_pred: {}'.format(fpga_pred))
                executor.add_predictor(acc_type=AccType.FPGA, qos_level=qos_level)
                self.available_predictors[3] -= 1
            while fpga_pred < executor.num_predictor_types[AccType.FPGA.value-1 + qos_level*4]:
                # print('fpga_pred: {}'.format(fpga_pred))
                executor.remove_predictor_by_type(acc_type=AccType.FPGA.value, qos_level=qos_level)
                self.available_predictors[3] += 1
            new_assignment[3 + qos_level*4] = executor.num_predictor_types[AccType.FPGA.value-1 + qos_level*4]

        logging.debug('Old assignment for executor {}: {}'.format(assignment[0], assignment[1:]))
        logging.debug('New applied assignment: {}'.format(new_assignment))
        logging.debug('Remaining available predictors: {}'.format(self.available_predictors))

        return new_assignment


    def get_available_predictors(self):
        return self.available_predictors

    
    def evaluate_reward(self, K):
        '''
        RL-Cache version of reward: play the trace out K steps into the future,
        find the # of missed requests, and then roll back
        '''
        logging.debug('--- temp_simulator: Playing trace into the future to evaluate reward ---')
        
        if len(self.event_queue) == 0:
            logging.warn('WARN: Ran out of requests while evaluating reward.')

        # only copying partial event queue
        partial_event_queue = copy.deepcopy(self.event_queue[:K*2])
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
        logging.debug('Failed requests: {}'.format(temp_simulator.failed_requests))
        logging.debug('--- temp_simulator: Rolling back ---')
        reward = -temp_simulator.failed_requests
        return reward


    def process(self, event, clock):
        '''
        Process the given event according to its EventType.
        '''
        if event.type == EventType.START_REQUEST:
            logging.debug('Starting event {}. (Time: {})'.format(event.desc, clock))
            isi = event.desc
            if isi not in self.executors:
                self.add_executor(isi)
            # call executor.process_request() on relevant executor
            executor = self.executors[isi]
            end_time, qos_met = executor.process_request(event, clock, self.runtimes)

            # we report QoS fail whenever (i) request fails, (ii) request succeeds but QoS is not met
            if not qos_met:
                self.qos_stats[self.isi_to_idx[isi], event.qos_level*2+1] += 1
            if end_time is None:
                self.failed_requests += 1
                self.failed_requests_arr[self.isi_to_idx[isi]] += 1
                self.qos_stats[self.isi_to_idx[isi], event.qos_level*2+1] += 1
                logging.debug('WARN: Request id {} for {} failed. (Time: {})'.format(event.id, event.desc, clock))
            else:
                # TODO: look into dequeue
                self.insert_event(end_time, EventType.END_REQUEST, event.desc, id=event.id, qos_level=event.qos_level)
            self.total_requests_arr[self.isi_to_idx[isi]] += 1
            self.qos_stats[self.isi_to_idx[isi], event.qos_level*2] += 1

        elif event.type == EventType.END_REQUEST:
            logging.debug('Event {} ended. (Time: {})'.format(event.desc, event.start_time))
            isi = event.desc
            executor = self.executors[isi]
            executor.finish_request(event, clock)

        elif event.type == EventType.SCHEDULING:
            sched_decision = self.invoke_scheduling_agent()
            if sched_decision is not None:
                logging.debug(sched_decision)
            else:
                logging.error('ERROR: The scheduling agent returned an exception. (Time: {})'.format(event.start_time))
            self.insert_event(clock+self.sched_interval, EventType.SCHEDULING, event.desc)

        return None


    def add_executor(self, isi):
        executor = Executor(isi, self.n_qos_levels)
        self.executors[executor.isi] = executor
        return executor.id


    def insert_event(self, time, type, desc, runtime=None, id='', deadline=100, qos_level=0):
        if type == EventType.END_REQUEST:
            event = Event(time, type, desc, runtime, id=id, deadline=deadline, qos_level=qos_level)
        else:
            event = Event(time, type, desc, runtime, deadline=deadline, qos_level=qos_level)

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
            mid = (start + end)//2
            if arr[mid].start_time == number:
                return mid
            elif arr[mid].start_time < number:
                start = mid + 1
            else:
                end = mid-1
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
            print('Time: {}, event {} of type {}'.format(event.start_time, event.desc, event.type))


class EventType(Enum):
    START_REQUEST = 1
    SCHEDULING = 2
    END_REQUEST = 3


class Event:
    def __init__(self, start_time, type, desc, runtime=None, deadline=100, id='', qos_level=0):
        self.id = id
        if self.id == '':
            self.id = uuid.uuid4().hex
        self.type = type
        self.start_time = start_time
        self.desc = desc
        self.runtime = runtime
        self.deadline = deadline
        self.qos_level = qos_level
    



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