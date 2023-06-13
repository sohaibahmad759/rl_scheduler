import os
import logging
import time
import traceback
import uuid
import pprint
import numpy as np
from enum import Enum

import sys
sys.path.append('../')
from core.common import Event, EventType


# TODO: the number of accelerator types should be parameterizable
#       we will potentially have 6 accelerator types, but it should be changeable
class AccType(Enum):
    CPU = 1
    GPU = 2
    VPU = 3
    FPGA = 4

    def __lt__(self, other):
        if self.__class__ is other.__class__:
           return self.value < other.value
        else:
            return NotImplemented


class ClusterPredictor:
    def __init__(self, logging_level, acc_type=AccType.CPU, qos_level=0,
                 profiled_accuracy=100.0, profiled_latencies={}, variant_name=None,
                 executor=None, simulator=None, configured_max_batch_size=None,
                 batching_algo=None, task_assignment=None, max_batch_size=None,
                 model_assignment=None, model=None, input_batch_1=None,
                 input_batch_2=None, input_batch_4=None, input_batch_8=None,
                 filename=None):
        # attributes related to predictor hardware
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging_level)
        self.logging_level = logging_level

        self.id = uuid.uuid4().hex
        self.acc_type = acc_type
        self.variant_name = variant_name
        self.qos_level = qos_level
        self.profiled_accuracy = profiled_accuracy
        self.profiled_latencies = profiled_latencies

        # attributes related to current status
        self.busy = False
        self.busy_till = None
        self.request_dict = {}
        self.clock = 0

        # Batching-related variables (by default we have a batch size of 1)
        self.event_queue = []
        self.request_queue = []
        self.event_counter = 0
        self.slo_expiring_dict = {}
        self.expiring_waiting = False

        self.load = None

        # self.executor = executor
        # self.simulator = simulator

        self.batch_sizes_allowed = [1, 2, 4, 8]
        self.configured_max_batch_size = configured_max_batch_size
        self.max_batch_size = max_batch_size
        if max_batch_size is None:
            raise Exception('max_batch_size is not configured')

        self.served_requests_per_step = 0
        self.incoming_requests_per_step = 0
        # if self.max_batch_size == 0:
        #     self.peak_throughput = 0
        # else:
        #     self.peak_throughput = self.max_batch_size * 1000 / profiled_latencies[(self.executor.isi,
        #                                                                             variant_name,
        #                                                                             self.max_batch_size)] 

        self.batching_algo = batching_algo
        if batching_algo is None:
            raise Exception('batching_algo not specified')
        # Only needed if model assignment and job scheduling policies are INFaaS v2
        self.infaas_batch_size = self.max_batch_size
        self.infaas_cost = np.inf
        # self.set_infaas_cost()
        self.aimd_batch_size = 1

        self.batch_expiring_set = False

        # self.task_assignment = self.executor.task_assignment
        self.task_assignment = task_assignment
        if task_assignment is None:
            raise Exception('task_assignment not specified')
        self.model_assignment = model_assignment
        if model_assignment is None:
            raise Exception('model_assignment not specified')

        # traceback.print_stack()
        # time.sleep(1)

        self.model = model
        self.input_batch_1 = input_batch_1
        self.input_batch_2 = input_batch_2
        self.input_batch_4 = input_batch_4
        self.input_batch_8 = input_batch_8
        
        # If the maximum batch size is 0, that means that predictor cannot even
        # serve a batch size of 1 without violating latency SLO
        if self.max_batch_size == 0:
            # if self.simulator.model_assignment != 'clipper':
            self.busy = True
            # if not('infaas' in self.simulator.model_assignment):
            #     raise PredictorException(f'Predictor {self.id} cannot be used as it will exceed latency SLO, '
            #         f'model variant: {self.variant_name}, accelerator type: '
            #         f'{self.acc_type}')
        
        # predictor_log = logging.FileHandler(f'logs/per_predictor/300ms/'
        #                                          f'{self.simulator.model_assignment}/'
        #                                          f'{self.id}.txt')
        # predictor_log.setLevel(logging.INFO)
        # self.predictor_log = logging.getLogger(self.id)
        # self.predictor_log.addHandler(predictor_log)

        if filename is None:
            raise Exception('filename is not given')
        
        if os.path.exists(f'cluster_logs/{filename}'):
            raise Exception(f'file already exists: cluster_logs/{filename}')
        
        filehandler = logging.FileHandler(f'cluster_logs/{filename}')
        filehandler.setLevel(logging.DEBUG)
        self.predictor_log = logging.getLogger(self.id)
        self.predictor_log.addHandler(filehandler)
        self.predictor_log.warn(f'variant name: {self.variant_name}, acc_type: {self.acc_type}')

        return
    

    def append_to_event_queue(self, event):
        self.event_queue.append(event)

    
    def set_load(self, load):
        self.load = load
        return
    

    def run_main_loop(self):
        self.predictor_log.warn('running main loop')
        while len(self.event_queue) > 0:
            event = self.event_queue.pop(0)
            # self.predictor_log.warn(f'event: {event.start_time}, type: {event.type}')
            if event.type == EventType.START_REQUEST:
                self.enqueue_request(event, event.start_time)
            elif event.type == EventType.FINISH_BATCH:
                self.finish_batch_callback(event.start_time, event.late)

    
    def increase_aimd_batch_size(self):
        ''' Since we only allow multiple of 4 batch sizes after 8, AIMD can only
         go one step up to the next available batch size
        '''
        # self.simulator.aimd_stats['increased'] += 1
        self.log.debug(f'increase_aimd_batch_size: current batch size: {self.aimd_batch_size}')
        current_idx = self.batch_sizes_allowed.index(self.aimd_batch_size)
        if current_idx < len(self.batch_sizes_allowed) - 1:
            new_idx = current_idx + 1
            self.aimd_batch_size = self.batch_sizes_allowed[new_idx]
            current_idx = new_idx
        # if self.aimd_batch_size > self.max_batch_size and self.max_batch_size > 0:
        #     self.aimd_batch_size = self.max_batch_size
        self.log.debug(f'increase_aimd_batch_size: batch size set: {self.aimd_batch_size}')
        return

    
    def decrease_aimd_batch_size(self):
        ''' Since we only allow multiple of 4 batch sizes after 8, AIMD can only
         go one step down to the previous available batch size
        '''
        # self.simulator.aimd_stats['decreased'] += 1
        self.log.debug(f'decrease_aimd_batch_size: current batch size: {self.aimd_batch_size}')
        current_idx = self.batch_sizes_allowed.index(self.aimd_batch_size)
        if current_idx > 0:
            new_idx = current_idx - 1
            self.aimd_batch_size = self.batch_sizes_allowed[new_idx]
            current_idx = new_idx
        self.log.debug(f'decrease_aimd_batch_size: batch size set: {self.aimd_batch_size}')
        return

    
    def get_infaas_cost(self):
        return self.infaas_cost

    
    def set_infaas_cost(self):
        ''' The cost of this model variant is the drop in accuracy when compared
        to the most accurate model in the model family
        '''
        all_variant_accuracies = self.executor.variant_accuracies
        model_name = self.executor.isi
        model_variant_accuracies = dict(filter(lambda x: x[0][0]==model_name,
                                               all_variant_accuracies.items()))

        highest_accuracy = max(model_variant_accuracies.values())
        accuracy_drop = highest_accuracy - self.profiled_accuracy
        self.infaas_cost = accuracy_drop
        return

    
    def get_infaas_batch_size(self):
        return self.infaas_batch_size

    
    def set_infaas_batch_size(self, batch_size):
        self.infaas_batch_size = batch_size
        return

    
    def get_largest_batch_size(self):
        largest_batch_sizes = self.simulator.get_largest_batch_sizes()

        acc_type = self.acc_type
        if acc_type == 1:
            acc_type = 'CPU'
        elif acc_type == 2:
            acc_type = 'GPU_AMPERE'
        elif acc_type == 3:
            acc_type = 'VPU'
        elif acc_type == 4:
            acc_type = 'GPU_PASCAL'
        
        largest_batch_size = largest_batch_sizes[(acc_type, self.variant_name)]
        
        # If self.configured_max_batch_size is None, there is no maximum batch size
        # specified, so we can use the largest batch size for the given accelerator
        # Otherwise, we need to cap it by self.configured_max_batch_size which is a
        # configuration parameter
        if self.configured_max_batch_size is not None:
            largest_batch_size = min(largest_batch_size, self.configured_max_batch_size)
        return largest_batch_size
    

    def assign_request(self, event, clock):
        # If request can be finished within deadline, return end_time, else return None (failed request)
        
        # TODO: request runtime should not be fixed, it should be based on profiled data
        if self.busy:
            end_time = self.busy_till + event.runtime
        else:
            end_time = clock + event.runtime
        # print(event.start_time + event.deadline)
        # print(event.deadline)
        if end_time <= event.start_time + event.deadline:
            self.busy = True
            self.busy_till = end_time
            self.request_dict[event.id] = 1
            # self.request_dict.append(event.id)
            return end_time
        else:
            # print('could not assign request')
            return None

    
    def finish_request(self, event):
        if event.id not in self.request_dict:
            return False
        
        # self.request_dict.remove(event.id)
        self.served_requests_per_step += 1
        del self.request_dict[event.id]
        if len(self.request_dict) == 0:
            # self.busy = False
            self.busy_till = None

        return True

    
    def enqueue_request(self, event, clock):
        ''' Add the request to the request queue of this predictor
        '''
        self.predictor_log.warn(f'enqueued,{clock}')
        # self.request_dict[event.id] = 1
        self.request_queue.append(event)
        self.event_counter += 1
        self.incoming_requests_per_step += 1

        # If predictor is busy, we have to wait until we get a FINISH_BATCH event
        # before we further process this request
        if self.busy:
            if self.batching_algo in ['aimd', 'nexus', 'infaas']:
                if self.batch_expiring_set == False:
                    if self.max_batch_size == 0:
                        # self.simulator.bump_failed_request_stats(event)
                        self.bump_failed_request_stats(event)
                        return
                    if self.batching_algo == 'nexus':
                        batch_expiring_set = clock + event.deadline - self.batch_processing_latency(self.max_batch_size, event)
                    elif self.batching_algo == 'aimd':
                        batch_expiring_set = clock + event.deadline - self.batch_processing_latency(self.aimd_batch_size, event)
                    elif self.batching_algo == 'infaas':
                        batch_expiring_set = clock + event.deadline - self.batch_processing_latency(self.infaas_batch_size, event)
                    self.generate_batch_expiring(event, batch_expiring_set)
                    return
            else:
                self.generate_head_slo_expiring()
            return

        # if self.task_assignment == TaskAssignment.INFAAS:
        #     if len(self.request_queue) >= self.infaas_batch_size:
        #         self.process_batch(clock, self.infaas_batch_size)
                
        # elif self.task_assignment == TaskAssignment.CANARY:

        # If we are past t_w(q), execute queue with q-1 requests
        # If q-1 is 0 at this point, the request deadlines are not set properly
        # and request cannot possibly be executed within deadline

        if self.task_assignment == 'canary' and self.batching_algo == 'accscale':
            self.pop_while_first_expires(clock)
            batch_size = self.find_batch_size(requests=len(self.request_queue))
            self.log.debug(f'enqueue_request: Trying to find appropriate batch size. Number '
                           f'of requests in queue: {len(self.request_queue)}, batch size '
                           f'returned: {batch_size}')
        elif self.task_assignment == 'canary' and self.batching_algo == 'aimd':
            # if len(self.request_queue) >= self.aimd_batch_size:
            #     # print(f'Calling process_batch from enqueue_request')
            #     # self.process_batch(clock, self.aimd_batch_size)
            #     batch_size = self.aimd_batch_size
            #     # self.increase_aimd_batch_size()
            # else:
            #     return
            batch_size = self.aimd_batch_size
            if len(self.request_queue) >= self.aimd_batch_size:
                print(f'aimd calling process_batch from enqueue_request')
                self.process_batch(clock, self.aimd_batch_size)
            return
        elif self.task_assignment == 'canary' and self.batching_algo == 'nexus':
            # self.pop_while_first_expires(clock)
            if len(self.request_queue) >= self.max_batch_size:
                self.process_batch(clock, self.max_batch_size)
            return
        elif self.task_assignment == 'infaas':
            # self.pop_while_first_expires(clock)
            batch_size = self.infaas_batch_size
            if len(self.request_queue) >= batch_size:
                self.process_batch(clock, batch_size)
            return
        else:
            raise Exception(f'Unexpected combination, task assignment: {self.task_assignment}, '
                            f'batching algorithm: {self.batching_algo}')

        # if self.batching_algo == 'aimd':
        #     # # AIMD should not be calling this, as it does not have early dropping
        #     # self.pop_while_first_expires(clock)
        #     batch_size = self.aimd_batch_size
        #     if len(self.request_queue) >= self.aimd_batch_size:
        #         self.log.debug(f'aimd calling process_batch from enqueue_request')
        #         self.process_batch(clock, self.aimd_batch_size)
        #     return

        if batch_size == -1:
            # requests in queue exceed maximum batch size
            self.process_batch(clock, self.max_batch_size)
        else:
            first_request = self.request_queue[0]
            first_request_expiration = first_request.start_time + first_request.deadline

            # TODO: this is only for debugging purposes. Remove when debugging complete
            if clock > first_request_expiration:
                raise Exception('Expired request has not been removed from the queue')

            if self.batch_processing_latency(1, first_request) > first_request.deadline:
                if 'infaas' in self.simulator.model_assignment:
                    self.simulator.bump_failed_request_stats(first_request)
                else:
                    raise Exception(f'Request cannot be processed even with batch size '
                                    f'of 1. deadline: {first_request.deadline}, processing '
                                    f'latency: {self.batch_processing_latency(1, first_request)}')

            max_waiting_time = first_request_expiration - self.batch_processing_latency(batch_size, first_request)

            # TODO: whenever a request is popped from the queue and the queue has a new head,
            #       we have to generate an SLO expiring event for it

            if clock < max_waiting_time:
                # we can still wait with new request in queue
                self.generate_slo_expiring(first_request, max_waiting_time)
                self.log.debug(f'Generated SLO_EXPIRING event for request {event.desc} '
                               f'to expire at {max_waiting_time}')
            else:
                # if we execute a batch with the latest request, we will miss SLO
                # of first request in the queue. therefore, execute batch size q-1 requests
                if self.batching_algo == 'accscale':
                    batch_size = self.find_batch_size(requests=len(self.request_queue)-1)
                elif self.task_assignment == 'infaas':
                    batch_size = self.infaas_batch_size
                else:
                    raise Exception(f'Unexpected combination, task assignment: '
                                    f'{self.task_assignment}, batching algorithm: '
                                    f'{self.batching_algo}')
                self.log.debug(f'Calling process batch from enqueue_request')
                self.process_batch(clock, batch_size)
                return

        return
    

    def bump_failed_request_stats(self, event):
        ''' Not yet implemented
        '''
        return

    
    def process_batch(self, clock, batch_size):
        ''' Dequeue the first `batch_size` requests from the queue and process
        them in a batch.
        '''
        # self.predictor_log.warn(f'process_batch,{clock},{batch_size}')

        if self.busy:
            raise Exception('process_batch called when predictor is busy')
        
        # if 'ilp' not in self.simulator.model_assignment or 'accscale' not in self.simulator.batching_algo:
        model_asn = self.model_assignment
        if 'ilp' in model_asn or 'infaas' in model_asn:
            self.drop_expired_requests(clock)
        
        # self.simulator.batch_size_counters[batch_size] += 1

        self.log.debug(f'process_batch called with batch size of {batch_size}')
        # time.sleep(1)

        self.log.debug(f'Requests in queue before popping: {len(self.request_queue)}')

        if batch_size == -1:
            self.log.error(f'process_batch received batch size of -1')
            time.sleep(10)

        # if self.max_batch_size == 0:
        #     batch_size = 1

        if batch_size > self.max_batch_size:
            # if self.batching_algo != 'aimd' and self.task_assignment != TaskAssignment.INFAAS: 
            #     self.log.error(f'process_batch received batch size of {batch_size}, max '
            #                 f'batch size: {self.max_batch_size}')   
            #     time.sleep(10)
            # else:
            #     batch_size = self.max_batch_size
            # self.predictor_log.warn(f'reducing batch size from {batch_size} to {self.max_batch_size}')
            batch_size = self.max_batch_size

        temp_queue = []
        dequeued_requests = 0

        while dequeued_requests < batch_size and len(self.request_queue) > 0:
            temp_queue.append(self.request_queue.pop(0))
            dequeued_requests += 1

        if dequeued_requests == 0:
            return

        # Since requests have been popped from the queue, we need to generate an
        # SLO_EXPIRING event for the new request at the head of the queue
        if len(self.request_queue) > 0:
            self.generate_head_slo_expiring()

        self.log.debug(f'Batch size given: {batch_size}, requests in queue after popping: '
                       f'{len(self.request_queue)}, dequeued_requests: {dequeued_requests}')

        batch_processing_time = self.batch_processing_latency(batch_size, temp_queue[0])
        finish_time = clock + batch_processing_time
        accuracy_seen = self.profiled_accuracy

        # TODO: handle qos_met appropriately; for now it is always set to True
        #       if request is processed within deadline
        qos_met = True

        aimd_negative_feedback = False
        execution_batch_size = len(temp_queue)

        execution_start_time = time.time()

        if execution_batch_size > 2 and execution_batch_size < 4:
            execution_batch_size = 2
        elif execution_batch_size > 4 and execution_batch_size < 8:
            execution_batch_size = 4

        if execution_batch_size == 1:
            input_batch = self.input_batch_1
        elif execution_batch_size == 2:
            input_batch = self.input_batch_2
        elif execution_batch_size == 4:
            input_batch = self.input_batch_4
        elif execution_batch_size == 8:
            input_batch = self.input_batch_8
        else:
            raise Exception(f'unexpected queue size: {execution_batch_size}')

        # if 't5' in self.variant_name or 'gpt2' in self.variant_name:
        if 't5' in self.variant_name:
            self.model.generate(input_batch)
        elif 'resnet' in self.variant_name or 'mobilenet' in self.variant_name or 'efficientnet' in self.variant_name:
            self.model(**input_batch)
        else:
            self.model(input_batch)

        execution_end_time = time.time()
        execution_time = (execution_end_time - execution_start_time) * 1000
        print(f'clock: {clock}, execution_time: {execution_time}, batch_size: {execution_batch_size}')

        if clock + execution_time > temp_queue[0].start_time + temp_queue[0].deadline:
            aimd_negative_feedback = True

        # for request in temp_queue:
        #     # self.log.error(f'request: {request}')
        #     # self.log.error(f'finish_time: {finish_time}, request.start_time: {request.start_time}, '
        #                 #    f'request.deadline: {request.deadline}')
        #     if finish_time > request.start_time + request.deadline:
        #         if self.task_assignment == 'canary':
        #             if self.batching_algo == 'accscale':
        #                 pass
        #                 # raise PredictorException(f'process_batch: Something is wrong, first request '
        #                 #                          f'in queue will expire before batch finishes processing '
        #                 #                          f'for batching algo: {self.batching_algo}')
        #             # # AIMD performs lazy-dropping
        #             elif self.batching_algo == 'aimd':
        #                 aimd_negative_feedback = True
        #                 pass
        #                 # self.simulator.bump_failed_request_stats(request)
        #                 # continue
        #             # Since Nexus performed early-drop, no need to drop here
        #             elif self.batching_algo == 'nexus':
        #                 pass
        #                 # self.simulator.bump_failed_request_stats(request)
        #                 # continue
        #             else:
        #                 raise Exception(f'unexpected batching algorithm: {self.batching_algo}')
        #         elif self.task_assignment == 'infaas':
        #             pass
        #             # self.simulator.bump_failed_request_stats(request)
        #             # continue
        #     self.generate_end_request_event(request, finish_time, accuracy_seen, qos_met)
        # self.generate_finish_batch_event(finish_time=finish_time,
        #                                  predictor=self,
        #                                  executor=None)
        
        if aimd_negative_feedback:
            self.decrease_aimd_batch_size()
        else:
            self.increase_aimd_batch_size()

        self.busy = True
        self.busy_till = finish_time

        if self.batching_algo == 'aimd':
            self.generate_finish_batch_event(finish_time=clock+execution_time,
                                             predictor=self,
                                             executor=None,
                                             late=aimd_negative_feedback)
            # self.finish_batch_callback(clock+execution_time, aimd_negative_feedback)

        # self.predictor_log.warn(f'end of process_batch, execution batch size: {execution_batch_size}, '
        #                         f'request queue size: {len(self.request_queue)}')
        self.predictor_log.warn(f'process_batch,{clock},{execution_batch_size}')

        return
    
    
    def generate_end_request_event(self, request, finish_time, accuracy_seen, qos_met):
        return
    

    def generate_finish_batch_event(self, finish_time, predictor, executor, late):
        ''' Insert finish batch callback in event queue
        '''
        event = Event(start_time=finish_time, type=EventType.FINISH_BATCH,
                        desc='finish_batch', predictor=predictor, late=late
                      )
        idx = self.binary_find_index(self.event_queue, finish_time)
        self.event_queue.insert(idx, event)
        return
    
    def generate_batch_expiring_event(self, time, event, predictor, executor, event_counter):
        return
    

    def generate_slo_expiring_event(self, time, event, predictor, executor, event_counter):
        return

    
    def finish_batch_callback(self, clock, late=False):
        ''' Callback to handle a FINISH_BATCH event
        '''
        self.predictor_log.warn(f'finish_batch_callback,{clock},{late}')
        self.predictor_log.warn(f'request queue len: {len(self.request_queue)}')

        self.busy = False
        self.busy_till = None

        if len(self.request_queue) == 0:
            return

        if self.task_assignment == 'canary':
            if self.batching_algo == 'accscale':
                if self.expiring_waiting:
                    self.expiring_waiting = False
                    batch_size = self.find_batch_size(len(self.request_queue))
                    if batch_size == -1:
                        batch_size = self.max_batch_size
                    self.process_batch(clock, batch_size)
                else:
                    # self.pop_while_first_expires(clock)
                    # if len(self.request_queue) == 0:
                    #     return
                    # batch_size = self.find_batch_size(len(self.request_queue))
                    # if batch_size == -1:
                    #     batch_size = self.max_batch_size
                    # self.process_batch(clock, batch_size)
                    if len(self.request_queue) > self.max_batch_size:
                        self.process_batch(clock, self.max_batch_size)
                return
            elif self.batching_algo == 'aimd':
                if len(self.request_queue) >= self.aimd_batch_size:
                    self.log.debug(f'Calling process_batch from finish_batch_callback')
                    self.process_batch(clock, self.aimd_batch_size)
                    # self.increase_aimd_batch_size()
                    return
                elif len(self.request_queue) > 0 and len(self.request_queue) < self.aimd_batch_size:
                    first_request = self.request_queue[0]
                    first_request_deadline = first_request.deadline
                    batch_expiring_time = clock + first_request_deadline - self.batch_processing_latency(self.aimd_batch_size, first_request)
                    self.generate_batch_expiring(first_request, batch_expiring_time)
                    return
            elif self.batching_algo == 'nexus':
                self.batch_expiring_set = False
                # self.pop_while_first_expires(clock)
                if len(self.request_queue) >= self.max_batch_size:
                    self.process_batch(clock, self.max_batch_size)
                elif len(self.request_queue) > 0 and len(self.request_queue) < self.max_batch_size:
                    first_request = self.request_queue[0]
                    first_request_deadline = first_request.deadline
                    batch_expiring_time = clock + first_request_deadline - self.batch_processing_latency(self.max_batch_size, first_request)
                    batch_expiring_time = clock + first_request_deadline - self.batch_processing_latency(self.max_batch_size, first_request)
                    self.generate_batch_expiring(first_request, batch_expiring_time)
                return
            else:
                self.log.error(f'finish_batch_callback: Unexpected batching algo: {self.batching_algo}')
        elif self.task_assignment == 'infaas':
            self.log.debug(f'infaas batch size: {self.infaas_batch_size}')
            if len(self.request_queue) >= self.infaas_batch_size:
                self.log.debug(f'Calling process_batch from finish_batch_callback for INFaaS')
                self.process_batch(clock, self.infaas_batch_size)
        else:
            self.log.error(f'finish_batch_callback encountered unexpected task '
                          f'assignment algorithm: {self.task_assignment}')
            time.sleep(10)

        return

    
    def pop_while_first_expires(self, clock):
        ''' This function removes the first request in the queue if it would expire
        by executing all the requests in the queue in a batch or with the maximum
        batch size allowed, iteratively until we do not have any request in the queue
        that would expire
        '''
        if self.batching_algo in ['aimd']:
            raise Exception(f'{self.batching_algo} should not be using '
                            f'pop_while_first_expires() since it drops request using '
                            f'an assumption of SLO_EXPIRING events which are not '
                            f'used for this batching algorithm')
        
        # Assume first request is expiring
        first_request_expiring = True
        queued_requests = len(self.request_queue)

        while first_request_expiring and queued_requests > 0:
            first_request = self.request_queue[0]
            first_request_expiration = first_request.start_time + first_request.deadline

            batch_size = self.find_batch_size(queued_requests)
            if batch_size == -1:
                batch_size = self.max_batch_size
            batch_processing_time = self.batch_processing_latency(batch_size, first_request)

            self.log.debug(f'FINISH_BATCH callback: current time: {clock}, first request '
                           f'starts at {first_request.start_time} and expires at {first_request_expiration}, '
                           f'time to process it will be {batch_processing_time} with batch '
                           f'size of {batch_size}, requests in queue: {queued_requests}')

            popped = False

            if clock + batch_processing_time > first_request_expiration:
                failed_request = self.request_queue.pop(0)
                self.simulator.bump_failed_request_stats(failed_request)
                queued_requests = len(self.request_queue)
                # self.simulator.slo_timeouts['timeouts'] += 1
                # self.simulator.slo_timeouts['total'] += 1
                popped = True
                # time.sleep(10)
            else:
                first_request_expiring = False
            
        
        # Since requests have been popped from the queue, we need to generate an
        # SLO_EXPIRING event for the new request at the head of the queue
        self.generate_head_slo_expiring()

        return

    
    def generate_slo_expiring(self, event, time):
        ''' Call the simulator's handler for generating an SLO_EXPIRING
        event
        '''
        if event.id in self.slo_expiring_dict and self.slo_expiring_dict[event.id] == time:
            # We don't need to generate a new SLO_EXPIRING event since we already have
            # an event with the same expiration time (because batch size hasn't changed)
            return
        self.log.debug(f'Generating SLO_EXPIRING event for request {event.id} '
                      f'to expire at {time}')
        self.generate_slo_expiring_event(time,
                                         event,
                                         predictor=self,
                                         executor=None,
                                         event_counter=event.id)
        self.slo_expiring_dict[event.id] = time
        # TODO: remove this line if SLO_EXPIRING works correctly
        self.event_counter += 1
        return
    

    def generate_batch_expiring(self, event, time):
        ''' Call the simulator's handler for generating a BATCH_EXPIRING
        event
        '''
        self.generate_batch_expiring_event(time,
                                           event,
                                           predictor=self,
                                           executor=None,
                                           event_counter=event.id)
        self.batch_expiring_set = True
        return

    
    def slo_expiring_callback(self, event, clock):
        ''' Callback to handle an SLO_EXPIRING event
        '''
        if self.busy is True:
            self.expiring_waiting = True
            return
        
        if self.batching_algo == 'nexus' or self.batching_algo == 'aimd' or self.batching_algo == 'infaas':
            self.log.debug(f'SLO expiring event encountered, ignoring for task assignment: '
                          f'{self.task_assignment}, batching algo: {self.batching_algo}')
            return

        self.log.debug(f'SLO expiring callback, Current clock: {clock}, event counter: '
            f'{event.event_counter}, slo_expiring_dict entry: {self.slo_expiring_dict[event.event_counter]}')
        if clock != self.slo_expiring_dict[event.event_counter]:
            # this means we have encountered an older SLO_EXPIRING callback
            # we do not need to handle it
            self.log.debug(f'slo_expiring_callback: Encountered an older SLO_EXPIRING '
                f'event for request {event.event_counter}. Latest value: '
                f'{self.slo_expiring_dict[event.event_counter]}, Value in '
                f'current event: {clock}')
            # time.sleep(5)
            return

        if len(self.request_queue) > 0 and self.request_queue[0].id != event.event_counter:
            self.log.debug(f'Request at head of queue: {self.request_queue[0].id}, SLO_EXPIRING event '
                           f'received for request: {event.event_counter}. Previous request: '
                           f'{(event.event_counter in self.slo_expiring_dict)}')
            # time.sleep(5)
            return

        if len(self.request_queue) == 0:
            # We might have already processed the request before reaching its
            # SLO EXPIRING
            return

        if self.task_assignment == 'canary' and self.batching_algo == 'accscale':
            batch_size = self.find_batch_size(requests=len(self.request_queue))
            self.log.debug(f'slo_expiring_callback: Trying to find appropriate batch size. '
                           f'Number of requests in queue: {len(self.request_queue)}, '
                           f'batch size returned: {batch_size}')
        elif self.task_assignment == 'canary' and self.batching_algo == 'aimd':
            batch_size = self.aimd_batch_size
        elif self.task_assignment == 'infaas':
            batch_size = self.infaas_batch_size
            if batch_size == 0:
                batch_size = 1
        
        if batch_size == -1:
            # requests in queue exceed maximum batch size, this should not happen
            # since we have not added any new requests
            # actually, this can happen in two cases:
            # 1. if we added new requests while the predictor was busy processing another batch
            # 2. if the requests in batch were already more than max batch size when head
            #   SLO was generated
            self.log.error('slo_expiring_callback: Something is wrong, find_batch_size returned -1')
            # time.sleep(10)
            batch_size = self.max_batch_size
            return
        
        # if self.batching_algo is not 'aimd':
        self.log.debug(f'Calling process_batch from slo_expiring_callback')
        self.process_batch(clock, batch_size)
        return
    

    def batch_expiring_callback(self, event, clock):
        if self.batching_algo == 'nexus':
            self.nexus_expiring_callback(event, clock)
        elif self.batching_algo == 'aimd':
            self.aimd_expiring_callback(event, clock)
        elif self.batching_algo == 'infaas':
            self.infaas_expiring_callback(event, clock)
        else:
            raise Exception(f'Unexpected batching algo: {self.batching_algo}')
    
    
    def nexus_expiring_callback(self, event, clock):
        ''' Callback to handle a Nexus BATCH_EXPIRING event
        '''
        if self.busy is True:
            return

        self.pop_while_first_expires(clock)

        if len(self.request_queue) == 0:
            self.log.debug(f'no requests in queue, returning from nexus expiring callback')
            return
        
        if len(self.request_queue) >= self.max_batch_size:
            batch_size = self.max_batch_size
        else:
            batch_size = self.find_batch_size(len(self.request_queue))
        # batch_size = min(len(self.request_queue), self.max_batch_size)
        self.process_batch(clock, batch_size)
        return
    

    def aimd_expiring_callback(self, event, clock):
        ''' Callback to handle an AIMD BATCH_EXPIRING event
        '''
        # AIMD does not respond to BATCH_EXPIRING. If this functionality is changed,
        # this function will be implemented similar to nexus_expiring_callback()
        pass


    def infaas_expiring_callback(self, event, clock):
        ''' Callback to handle an INFaaS BATCH_EXPIRING event
        '''
        if self.busy is True:
            return
        
        # self.pop_while_first_expires(clock)
        
        if len(self.request_queue) == 0:
            return
        
        batch_size = min(self.infaas_batch_size, self.find_batch_size(len(self.request_queue)))
        # batch_size = self.find_batch_size(len(self.request_queue)))
        if batch_size == -1:
            batch_size = self.infaas_batch_size

        self.process_batch(clock, batch_size)
        return
    

    def drop_expired_requests(self, clock):
        ''' Drop all expired requests from the queue
        '''
        drop_indices = []
        for i in range(len(self.request_queue)):
            request = self.request_queue[i]
            if clock > request.start_time + request.deadline * 2:
                drop_indices.append(i)

        dropped = 0
        for i in range(len(drop_indices)):
            drop_idx = drop_indices[i]
            request = self.request_queue.pop(drop_idx-dropped)
            # print(f'we want to remove this from the event queue as well')
            # time.sleep(1)
            self.bump_failed_request_stats(request)
            dropped += 1

        drop_indices = []
        for i in range(len(self.event_queue)):
            request = self.event_queue[i]
            if clock > request.start_time + request.deadline * 2:
                drop_indices.append(i)

        dropped = 0
        for i in range(len(drop_indices)):
            drop_idx = drop_indices[i]
            request = self.event_queue.pop(drop_idx-dropped)
            dropped += 1
        return

    
    def generate_head_slo_expiring(self):
        ''' If any request is popped from the head of the queue and the queue has
        a new head, we need to generate an SLO_EXPIRING event for it to keep track
        of its expiry
        '''
        if len(self.request_queue) == 0:
            return

        first_request = self.request_queue[0]
        first_request_expiration = first_request.start_time + first_request.deadline

        while self.batch_processing_latency(batch_size=1, request=first_request) > first_request.deadline:
            # if 'infaas' in self.simulator.model_assignment or 'accscale' in self.simulator.model_assignment:
            if 'accscale' in self.simulator.model_assignment or 'accscale' in self.simulator.batching_algo:
                self.simulator.bump_failed_request_stats(first_request)
                self.request_queue.pop(0)
                if len(self.request_queue) == 0:
                    return
                first_request = self.request_queue[0]
                first_request_expiration = first_request.start_time + first_request.deadline
            else:
                raise Exception(f'Request cannot be processed even with batch size of 1')
        
        # TODO: what if we are already past the t_w for this batch size?
        batch_size = self.find_batch_size(requests=len(self.request_queue))

        if batch_size == -1:
            batch_size = self.max_batch_size

        max_waiting_time = first_request_expiration - self.batch_processing_latency(batch_size, first_request)

        self.generate_slo_expiring(first_request, max_waiting_time)
        self.log.debug(f'head: Generated SLO_EXPIRING event for request {first_request.desc} '
                      f'to expire at {max_waiting_time}')
        return

    
    def batch_processing_latency(self, batch_size, request):
        ''' Return the latency to process a batch of a given size
        '''
        self.log.debug('batch_processing_latency()')
        self.log.debug(f'Profiled latencies: {self.profiled_latencies}')
        self.log.debug(f'Request desc: {request.desc}, qos_level: {request.qos_level}, '
                      f'profiled latency: {self.profiled_latencies[(request.desc, self.variant_name, batch_size)]}')
        # processing_latency = self.profiled_latencies[(request.desc, self.variant_name, batch_size)] * batch_size
        processing_latency = self.profiled_latencies[(request.desc, self.variant_name, batch_size)]
        # self.log.error(f'tuple: {(request.desc, self.variant_name, batch_size)}, profiled latency: {self.profiled_latencies[(request.desc, self.variant_name, batch_size)]}')
        # self.log.error(f'processing_latency: {processing_latency}')
        # if self.acc_type == 4:
        #     print(f'predictor type: {self.acc_type}, self.profiled_latencies:')
        #     pprint.pprint(self.profiled_latencies)
        return processing_latency

    
    def find_batch_size(self, requests):
        ''' Find the appropriate batch size for a given number of requests by
        rounding up to the nearest bigger batch size
        '''
        # TODO: make sure that binary_search_index never exceeds self.max_batch_size
        # batch_size_index = self.binary_search_index(self.batch_sizes_allowed, requests)
        batch_size_index = self.find_maximum_that_fills(requests)
        if batch_size_index >= len(self.batch_sizes_allowed):
            return -1
        else:
            batch_size = self.batch_sizes_allowed[batch_size_index]
            return batch_size

    
    def find_maximum_that_fills(self, requests):
        ''' Alternate way of finding an appropriate batch size. We select a
        batch size that gets completely filled up by the current queue,
        instead of rounding up to the nearest bigger batch size
        '''
        if requests == 0:
            return -1

        idx = 0
        batch_idx = 0
        while idx < len(self.batch_sizes_allowed):
            # We cannot exceed the maximum batch size in our search
            if self.batch_sizes_allowed[idx] > self.max_batch_size:
                return batch_idx

            if requests >= self.batch_sizes_allowed[idx]:
                batch_idx = idx
            else:
                return batch_idx
            idx += 1
        return batch_idx

    
    def binary_search_index(self, arr, number):
        ''' Finds the appropriate batch size for a given number of requests
        by rounding up to the nearest bigger batch size available.
        For example, if we support batch sizes of 2 and 4, a queue of 3
        requests will be given a batch size of 4 according to this.
        Note: This might result in lower throughput as we are increasing
        latency but finishing a smaller number of requests
        '''
        # Lower and upper bounds
        start = 0
        end = len(arr) - 1

        # Traverse the search space
        while start <= end:
            mid = (start + end) // 2
            if arr[mid] == number:
                return mid
            elif arr[mid] < number:
                start = mid + 1
            else:
                end = mid - 1
        # Return the insert position
        return end + 1
    
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
