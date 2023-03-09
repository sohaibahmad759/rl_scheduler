import logging
import time
import uuid
import pprint
import numpy as np
from enum import Enum
from core.common import TaskAssignment
from core.exceptions import PredictorException


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


class Predictor:
    def __init__(self, logging_level, acc_type=AccType.CPU, qos_level=0,
                 profiled_accuracy=100.0, profiled_latencies={}, variant_name=None,
                 executor=None, simulator=None):
        # attributes related to predictor hardware
        self.log = logging.getLogger(__name__)

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

        # Batching-related variables (by default we have a batch size of 1)
        self.request_queue = []
        self.event_counter = 0
        self.slo_expiring_dict = {}

        self.load = None

        self.executor = executor
        self.simulator = simulator

        # self.batch_sizes_allowed = [1]
        # self.batch_sizes_allowed = [1, 2, 4, 8]
        # self.batch_sizes_allowed = [1, 2, 4, 8, 16]
        # self.batch_sizes_allowed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.batch_sizes_allowed = self.simulator.allowed_batch_sizes
        self.max_batch_size = self.get_largest_batch_size()

        self.batching_algo = self.simulator.batching_algo
        # Only needed if model assignment and job scheduling policies are INFaaS v2
        self.infaas_batch_size = self.max_batch_size
        self.infaas_cost = np.inf
        self.set_infaas_cost()
        self.aimd_batch_size = 1

        self.task_assignment = self.executor.task_assignment
        
        # If the maximum batch size is 0, that means that predictor cannot even
        # serve a batch size of 1 without violating latency SLO
        if self.max_batch_size == 0:
            # if self.simulator.model_assignment != 'clipper':
            self.busy = True
            raise PredictorException(f'Predictor {self.id} cannot be used as it will exceed latency SLO, '
                f'model variant: {self.variant_name}, accelerator type: '
                f'{self.acc_type}')
            time.sleep(10)
        return

    
    def set_load(self, load):
        self.load = load
        return

    
    def increase_aimd_batch_size(self):
        ''' Since we only allow batch sizes of [1, 2, 4, 8], AIMD can only
         go one step up to the next available batch size
        '''
        self.log.debug(f'increase_aimd_batch_size: current batch size: {self.aimd_batch_size}')
        current_idx = self.batch_sizes_allowed.index(self.aimd_batch_size)
        if current_idx < len(self.batch_sizes_allowed) - 1:
            new_idx = current_idx + 1
            self.aimd_batch_size = self.batch_sizes_allowed[new_idx]
            current_idx = new_idx
        if self.aimd_batch_size > self.max_batch_size and self.max_batch_size > 0:
            self.aimd_batch_size = self.max_batch_size
        self.log.debug(f'increase_aimd_batch_size: batch size set: {self.aimd_batch_size}')
        return

    
    def decrease_aimd_batch_size(self):
        ''' Since we only allow batch sizes of [1, 2, 4, 8], AIMD can only
         go one step down to the previous available batch size
        '''
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
        model_variant_accuracies = dict(filter(lambda x: x[0][0]==model_name, all_variant_accuracies.items()))

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
        # self.simulator.bump_successful_request_stats(event)
        del self.request_dict[event.id]
        if len(self.request_dict) == 0:
            self.busy = False
            self.busy_till = None

        return True

    
    def enqueue_request(self, event, clock):
        ''' Add the request to the request queue of this predictor
        '''
        self.request_dict[event.id] = 1
        self.request_queue.append(event)
        self.event_counter += 1

        # If predictor is busy, we have to wait until we get a FINISH_BATCH event
        # before we further process this request
        if self.busy:
            self.generate_head_slo_expiring()
            return

        # if self.task_assignment == TaskAssignment.INFAAS:
        #     if len(self.request_queue) >= self.infaas_batch_size:
        #         self.process_batch(clock, self.infaas_batch_size)
                
        # elif self.task_assignment == TaskAssignment.CANARY:

        # If we are past t_w(q), execute queue with q-1 requests
        # If q-1 is 0 at this point, the request deadlines are not set properly
        # and request cannot possibly be executed within deadline

        if self.task_assignment == TaskAssignment.CANARY and self.batching_algo == 'accscale':
            batch_size = self.find_batch_size(requests=len(self.request_queue))
            self.log.debug(f'enqueue_request: Trying to find appropriate batch size. Number '
                           f'of requests in queue: {len(self.request_queue)}, batch size '
                           f'returned: {batch_size}')
        elif self.task_assignment == TaskAssignment.CANARY and self.batching_algo == 'aimd':
            if len(self.request_queue) >= self.aimd_batch_size:
                # print(f'Calling process_batch from enqueue_request')
                # self.process_batch(clock, self.aimd_batch_size)
                batch_size = self.aimd_batch_size
                self.increase_aimd_batch_size()
            else:
                return
        elif self.task_assignment == TaskAssignment.INFAAS:
            batch_size = self.infaas_batch_size
        else:
            raise PredictorException(f'Unexpected situation')

        if self.batching_algo == 'aimd':
            self.log.debug(f'AIMD calling pop_while_first_expires')
            self.pop_while_first_expires(clock)
            batch_size = self.aimd_batch_size
            if len(self.request_queue) == 0:
                return
            
            if len(self.request_queue) >= batch_size:
                self.log.debug(f'aimd calling process_batch from enqueue_request')
                self.process_batch(clock, batch_size)
                return
            else:
                return

        if batch_size == -1:
            # requests in queue exceed maximum batch size
            self.process_batch(clock, self.max_batch_size)
        else:
            first_request = self.request_queue[0]
            first_request_expiration = first_request.start_time + first_request.deadline

            # TODO: this is only for debugging purposes. Remove when debugging complete
            if clock > first_request_expiration:
                raise PredictorException('Expired request has not been removed from the queue')

            if self.batch_processing_latency(1, first_request) > first_request.deadline:
                raise PredictorException(f'Request cannot be processed even with batch size '
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
                elif self.batching_algo == 'aimd':
                    batch_size = self.aimd_batch_size
                elif self.task_assignment == TaskAssignment.INFAAS:
                    batch_size = self.infaas_batch_size
                self.log.debug(f'Calling process batch from enqueue_request')
                self.process_batch(clock, batch_size)
                return
        # else:
        #     logging.error(f'enqueue_request received unexpected task assignment '
        #                   f'algorithm: {self.task_assignment}')
        #     time.sleep(10)

        return

    
    def process_batch(self, clock, batch_size):
        ''' Dequeue the first `batch_size` requests from the queue and process
        them in a batch.
        '''
        # TODO: Are we making sure that none of the requests will be past deadline
        #       by the time the batch finishes processing?

        # TODO: Even after all the checks, make sure to track how many reqeusts
        #       finish after their deadline
        self.log.debug(f'process_batch called with batch size of {batch_size}')
        # time.sleep(1)

        self.log.debug(f'Requests in queue before popping: {len(self.request_queue)}')

        if batch_size == -1:
            self.log.error(f'process_batch received batch size of -1')
            time.sleep(10)

        # if self.max_batch_size == 0:
        #     batch_size = 1

        if batch_size > self.max_batch_size:
            self.log.error(f'process_batch received batch size of {batch_size}, max '
                           f'batch size: {self.max_batch_size}')
            if self.simulator.model_assignment != 'clipper' and self.task_assignment != TaskAssignment.INFAAS:    
                time.sleep(10)
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

        # self.log.error(f'going through temp_queue')
        for request in temp_queue:
            # self.log.error(f'request: {request}')
            # self.log.error(f'finish_time: {finish_time}, request.start_time: {request.start_time}, '
                        #    f'request.deadline: {request.deadline}')
            if finish_time > request.start_time + request.deadline:
                if self.task_assignment == TaskAssignment.CANARY:
                    if self.batching_algo == 'accscale':
                        raise PredictorException('process_batch: Something is wrong, first request in queue '
                            'will expire before batch finishes processing')
                        time.sleep(10)
                    elif self.batching_algo == 'aimd':
                        self.simulator.bump_failed_request_stats(request)
                        continue
                    else:
                        raise PredictorException('process_batch: unexpected situation')
                        time.sleep(10)
                elif self.task_assignment == TaskAssignment.INFAAS:
                    self.simulator.bump_failed_request_stats(request)
                    continue
            self.simulator.generate_end_request_event(request, finish_time,
                                                    accuracy_seen, qos_met)
        self.simulator.generate_finish_batch_event(finish_time=finish_time,
                                                predictor=self,
                                                executor=self.executor)

        self.busy = True
        self.busy_till = finish_time
        return

    
    def finish_batch_callback(self, clock):
        ''' Callback to handle a FINISH_BATCH event
        '''
        self.busy = False
        self.busy_till = None

        if self.task_assignment == TaskAssignment.CANARY:
            # If we run the batch with the requests currently in the queue, we want to
            # ensure that no request in the queue will expire by the time the batch
            # finishes processing

            # At this point, no request in the queue is past the point where it would
            # expire if executed in a batch
            # If we still have enough requests to fill the max batch size, we execute
            # the batch. Otherwise, we wait to get more requests or until we reach an
            # SLO_EXPIRING event
            self.pop_while_first_expires(clock)
            if self.batching_algo == 'accscale':
                if len(self.request_queue) >= self.max_batch_size:
                    self.log.debug(f'Calling process_batch from finish_batch_callback')

                    # approp = self.find_batch_size(len(self.request_queue))
                    # self.log.error(f'appropriate batch size: {approp}')
                    # first_request = self.request_queue[0]
                    # self.log.error(f'clock: {clock}, first request expires at: {(clock+first_request.deadline)}, '
                    #                f'batch processing latency: {self.batch_processing_latency(approp, first_request)}')

                    self.process_batch(clock, self.max_batch_size)
                    return
            elif self.batching_algo == 'aimd':
                if len(self.request_queue) >= self.aimd_batch_size:
                    self.log.debug(f'Calling process_batch from finish_batch_callback')
                    self.process_batch(clock, self.aimd_batch_size)
                    self.increase_aimd_batch_size()
                    return
            else:
                self.log.error(f'finish_batch_callback: Unexpected batching algo: {self.batching_algo}')
        elif self.task_assignment == TaskAssignment.INFAAS:
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
        # Assume first request is expiring
        first_request_expiring = True
        queued_requests = len(self.request_queue)
        # sorted_request_queue = sorted(self.request_queue)
        # if self.request_queue != sorted_request_queue:
        #     # self.log.error(f'request_queue start times: {list(map(lambda x: x.start_time, self.request_queue))}')
        #     # raise PredictorException('request queue is not sorted')
        #     self.log.warn(f'Request queue is not sorted, this should not be happening. '
        #                   f'For now a temporary fix has been made to sort the request '
        #                   f'queue every time a popping operation is done, but this may '
        #                   f'slow down code, although it should still be correct. If code '
        #                   f'is very slow with this but fast without sorting, find why this '
        #                   f'is happening and fix it.')
        #     self.request_queue = sorted_request_queue


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
                self.simulator.slo_timeouts['timeouts'] += 1
                self.simulator.slo_timeouts['total'] += 1
                popped = True
                # time.sleep(10)
            else:
                first_request_expiring = False
            
            if self.batching_algo == 'aimd' and popped:
                self.decrease_aimd_batch_size()
        
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
        self.simulator.generate_slo_expiring_event(time, event,
                                                predictor=self,
                                                executor=self.executor,
                                                event_counter=event.id)
        self.slo_expiring_dict[event.id] = time
        # TODO: remove this line if SLO_EXPIRING works correctly
        self.event_counter += 1
        return

    
    def slo_expiring_callback(self, event, clock):
        ''' Callback to handle an SLO_EXPIRING event
        '''
        # if self.task_assignment == TaskAssignment.INFAAS:
        #     logging.info('SLO expiring event encountered, ignoring for INFaaS')
        #     return

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

        if self.task_assignment == TaskAssignment.CANARY and self.batching_algo == 'accscale':
            batch_size = self.find_batch_size(requests=len(self.request_queue))
            self.log.debug(f'slo_expiring_callback: Trying to find appropriate batch size. '
                           f'Number of requests in queue: {len(self.request_queue)}, '
                           f'batch size returned: {batch_size}')
        elif self.task_assignment == TaskAssignment.CANARY and self.batching_algo == 'aimd':
            batch_size = self.aimd_batch_size
        elif self.task_assignment == TaskAssignment.INFAAS:
            batch_size = self.infaas_batch_size
        
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

    
    def generate_head_slo_expiring(self):
        ''' If any request is popped from the head of the queue and the queue has
        a new head, we need to generate an SLO_EXPIRING event for it to keep track
        of its expiry
        '''
        if len(self.request_queue) == 0:
            return

        first_request = self.request_queue[0]
        first_request_expiration = first_request.start_time + first_request.deadline

        if self.batch_processing_latency(batch_size=1, request=first_request) > first_request.deadline:
            raise PredictorException(f'Request cannot be processed even with batch size of 1')
            time.sleep(10)
        
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
        processing_latency = self.profiled_latencies[(request.desc, self.variant_name, batch_size)] * batch_size
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
