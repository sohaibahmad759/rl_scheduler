import itertools
import logging
import math
import random
import time
import uuid
import numpy as np
from enum import Enum
from predictor import AccType, Predictor


class Behavior(Enum):
    BESTEFFORT = 1
    STRICT = 2


class TaskAssignment(Enum):
    RANDOM = 1
    ROUND_ROBIN = 2
    EARLIEST_FINISH_TIME = 3
    LATEST_FINISH_TIME = 4
    INFAAS = 5


class Executor:
    def __init__(self, isi, task_assignment, n_qos_levels=1, behavior=Behavior.BESTEFFORT, runtimes=None,
                    variant_runtimes=None, variant_loadtimes=None):
        self.id = uuid.uuid4().hex
        self.isi = isi
        self.n_qos_levels = n_qos_levels
        self.predictors = {}
        self.num_predictor_types = np.zeros(4 * n_qos_levels)
        self.assigned_requests = {}
        self.iterator = itertools.cycle(self.predictors)
        self.behavior = behavior
        self.task_assignment = TaskAssignment(task_assignment)
        self.runtimes = runtimes

        self.variant_runtimes = variant_runtimes
        self.variant_loadtimes = variant_loadtimes
        self.variant_accuracies = {}

        self.model_variants = {}
        
        # EITHER: do we want a separate event queue for each executor? then we would need to
        # have another clock and interrupt when request ends
        # OR: a better way would be to just have predictors in the executor that we mark
        # busy when a request comes to them
        # if all predictors are busy, then the request either waits or has to fail
        # 4 ways to do this: round robin, earliest start time, earliest finish time (EFT), latest finish time


    def add_predictor(self, acc_type=AccType.CPU, qos_level=0):
        # print('acc_type: {}'.format(acc_type.value))
        profiled_latencies = self.runtimes[acc_type.value]
        predictor = Predictor(acc_type.value, qos_level=qos_level, profiled_accuracy=100.0,
                                profiled_latencies=profiled_latencies)
        self.predictors[predictor.id] = predictor
        self.num_predictor_types[acc_type.value-1 + qos_level*4] += 1
        self.iterator = itertools.cycle(self.predictors)
        return id


    def set_runtimes(self, runtimes=None):
        self.runtimes = runtimes

    
    def set_loadtimes(self, loadtimes=None):
        self.loadtimes = loadtimes

    
    def set_model_variants(self, model_variants={}):
        self.model_variants = model_variants


    def set_variant_accuracies(self, accuracies=None):
        self.variant_accuracies = accuracies

    
    def set_variant_runtimes(self, runtimes=None):
        self.variant_runtimes = runtimes

    
    def set_variant_loadtimes(self, loadtimes=None):
        self.variant_loadtimes = loadtimes

    
    def remove_predictor_by_id(self, id):
        if id in self.predictors:
            predictor_type = self.predictors[id].acc_type.value
            predictor_qos = self.predictors[id].qos_level
            self.num_predictor_types[predictor_type-1 + predictor_qos*4] -= 1
            del self.predictors[id]
            self.iterator = itertools.cycle(self.predictors)
            return True
        else:
            return False

    
    def remove_predictor_by_type(self, acc_type, qos_level=0):
        ''' If predictor of given type exists, remove it and return True.
            Otherwise, return False.
        '''
        for id in self.predictors:
            predictor_type = self.predictors[id].acc_type
            predictor_qos = self.predictors[id].qos_level
            if acc_type == predictor_type and qos_level == predictor_qos:
                self.num_predictor_types[predictor_type-1 + predictor_qos*4] -= 1
                del self.predictors[id]
                self.iterator = itertools.cycle(self.predictors)
                return True
        return False


    def process_request(self, event, clock, runtimes):
        if len(self.predictors) == 0:
            return None, False
            
        qos_met = True

        # Step 1: load balance

        # filter out predictors that match the request's QoS level
        filtered_predictors = list(filter(lambda key: self.predictors[key].qos_level >= event.qos_level, self.predictors))
        
        # TODO: is this the implementation of behavior we want? If there is any matching QoS level,
        #       it will be used regardless of finish time or any other metric
        # TODO: what to do if there are no predictors matching QoS level?
        if len(filtered_predictors) == 0:
            qos_met = False
            logging.debug('No predictors found for QoS level {} for request of {}'.format(event.qos_level, event.desc))
            if self.behavior == Behavior.BESTEFFORT:
                # choose from any QoS level
                filtered_predictors = list(self.predictors.keys())
            # TODO: increment missed QoS counter (histogram)

        if self.task_assignment == TaskAssignment.RANDOM:
            predictor = self.predictors[random.choice(filtered_predictors)]
        elif self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME or self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
            finish_times = []
            for key in filtered_predictors:
                candidate = self.predictors[key]
                potential_runtime = candidate.profiled_latencies[(event.desc, event.qos_level)]
                if candidate.busy:
                    finish_time = candidate.busy_till + potential_runtime
                else:
                    finish_time = clock + potential_runtime

                # finish_time has to be within the request's deadline
                if finish_time > clock + event.deadline:
                    # set it to an invalid value
                    if self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME:
                        finish_time = np.inf
                    elif self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
                        finish_time = -1
                finish_times.append(finish_time)
            logging.debug('filtered_predictors: {}'.format(filtered_predictors))
            logging.debug('finish_times: {}'.format(finish_times))
            if self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME:
                idx = finish_times.index(min(finish_times))
            elif self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
                idx = finish_times.index(max(finish_times))
            logging.debug('idx: {}'.format(idx))
            predictor = self.predictors[filtered_predictors[idx]]
            logging.debug('filtered_predictors[idx]: {}'.format(filtered_predictors[idx]))
            logging.debug('predictor: {}'.format(predictor))
            # time.sleep(2)
        elif self.task_assignment == TaskAssignment.INFAAS:
            accuracy_filtered_predictors = list(filter(lambda key: self.predictors[key].profiled_accuracy >= event.accuracy, self.predictors))
            predictor = None
            infaas_candidates = []
            not_found_reason = 'None'
            print()
            # There is atleast one predictor that matches the accuracy requirement of the request
            if len(accuracy_filtered_predictors) > 0:
                # If there is any predictor that can meet request
                for key in accuracy_filtered_predictors:
                    _predictor = self.predictors[key]
                    peak_throughput = math.floor(1000 /  _predictor.profiled_latencies[(event.desc, 
                                                    event.qos_level)])
                    queued_requests = len(_predictor.request_queue)

                    print('Throughput:', peak_throughput)
                    print('Queued requests:', queued_requests)
                    if peak_throughput > queued_requests:
                        _predictor.set_load(float(queued_requests)/peak_throughput)
                        infaas_candidates.append(_predictor)
                    else:
                        continue
                
                # We have now found a list of candidate predictors that match both accuracy
                # and deadline of the request, sorted by load
                infaas_candidates.sort(key=lambda x: x.load)

                # for candidate in infaas_candidates:
                #     print('infaas candidate:' + str(candidate) + ', load:' + str(candidate.load))
                # print('infaas candidates:' + str(infaas_candidates))
                # time.sleep(5)

                if len(infaas_candidates) > 0:
                    # Select the least loaded candidate
                    predictor = infaas_candidates[0]
                    logging.debug('found')
                else:
                    # There is no candidate that meets deadline
                    not_found_reason = 'latency_not_met'
                    logging.debug('not found')
            else:
                # There is no predictor that even matches the accuracy requirement of the request
                not_found_reason = 'accuracy_not_met'
                logging.debug('not found')
            
            # No predictor has been found yet, either because accuracy not met or deadline not met
            if predictor is None:
                print()
                # Now we try to find an inactive model variant that can meet accuracy+deadline
                isi_name = event.desc
                inactive_candidates = []
                checked_qos_levels = list(map(lambda key: self.predictors[key].qos_level, self.predictors))
                logging.debug('checked qos level:' + str(checked_qos_levels))

                print('model variants:' + str(self.model_variants))
                print('model variant accuracies:' + str(self.variant_accuracies))
                print('model variant runtimes:' + str(self.variant_runtimes))
                print('model variant loadtimes:' + str(self.variant_loadtimes))
                for qos_level in range(self.n_qos_levels):
                    # If there is no such variant (runtime is math.inf), skip
                    print(self.runtimes)
                    if math.isinf(self.runtimes[(isi_name, qos_level)]):
                        continue
                    # If we already checked for this variant
                    elif qos_level in checked_qos_levels:
                        continue
                    else:
                        runtime = self.runtimes[(isi_name, qos_level)]
                        loadtime = self.loadtimes[(isi_name, qos_level)]
                        total_time = runtime + loadtime
                        inactive_candidates[isi_name] = total_time
                        print('got here')
                        time.sleep(10)
                

                # If we still cannot find one, we try to serve with the closest possible accuracy and/or deadline

        # round-robin:
        # predictor = self.predictors[next(self.iterator)]

        # At this point, the variable 'predictor' should indicate which predictor has been
        # selected by one of the heuristics above

        # Step 2: read up runtime based on the predictor type selected
        runtime = runtimes[predictor.acc_type][(event.desc, event.qos_level)]
        event.runtime = runtime
        # print('executor.py -- Request runtime read from dict: {}'.format(runtime))

        # Step 3: assign to predictor selected
        assigned = predictor.assign_request(event, clock)
        if assigned is not None:
            self.assigned_requests[event.id] = predictor
            # print('self.assigned_requests: {}'.format(self.assigned_requests))
        else:
            logging.debug('WARN: Request id {} for {} could not be assigned to any predictor. (Time: {})'.format(event.id, event.desc, clock))

        return assigned, qos_met

    
    def finish_request(self, event, clock):
        # print('self.assigned_requests: {}'.format(self.assigned_requests))
        if event.id not in self.assigned_requests:
            logging.error('ERROR: Could not find assigned request.')
            return False
        
        predictor = self.assigned_requests[event.id]
        finished = predictor.finish_request(event)
        if finished:
            del self.assigned_requests[event.id]
            logging.debug('Finished request for {} at predictor {}. (Time: {})'.format(event.desc, predictor.id, clock))
            return True
        else:
            logging.debug('WARN: Could not finish request at predictor {}, \
                    executor {}. (Time: {})'.format(predictor.id, self.id, clock))
                    