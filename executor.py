import itertools
import logging
import random
import uuid
import numpy as np
from enum import Enum
from predictor import AccType, Predictor


class Behavior(Enum):
    BESTEFFORT = 1
    STRICT = 2


class Executor:
    def __init__(self, isi, n_qos_levels=1, behavior=Behavior.BESTEFFORT):
        self.id = uuid.uuid4().hex
        self.isi = isi
        self.n_qos_levels = n_qos_levels
        self.predictors = {}
        self.num_predictor_types = np.zeros(4 * n_qos_levels)
        self.assigned_requests = {}
        self.iterator = itertools.cycle(self.predictors)
        self.behavior = behavior
        # EITHER: do we want a separate event queue for each executor? then we would need to
        # have another clock and interrupt when request ends
        # OR: a better way would be to just have predictors in the executor that we mark
        # busy when a request comes to them
        # if all predictors are busy, then the request either waits or has to fail
        # 3 ways to do this: round robin, earliest start time, earliest finish time (EFT)

    def add_predictor(self, acc_type=AccType.CPU, qos_level=0):
        # print('acc_type: {}'.format(acc_type.value))
        predictor = Predictor(acc_type.value, qos_level=qos_level)
        self.predictors[predictor.id] = predictor
        self.num_predictor_types[acc_type.value-1 + qos_level*4] += 1
        self.iterator = itertools.cycle(self.predictors)
        return id

    
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
        filtered_predictors = list(filter(lambda key: self.predictors[key].qos_level == event.qos_level, self.predictors))
        
        # TODO: what to do if there are no predictors matching QoS level?
        if len(filtered_predictors) == 0:
            qos_met = False
            logging.debug('No predictors found for QoS level {} for request of {}'.format(event.qos_level, event.desc))
            if self.behavior == Behavior.BESTEFFORT:
                # choose from any QoS level
                filtered_predictors = list(self.predictors.keys())
            # TODO: increment missed QoS counter (histogram)

        # for now, we use random
        predictor = self.predictors[random.choice(filtered_predictors)]

        # round-robin:
        # predictor = self.predictors[next(self.iterator)]

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
                    