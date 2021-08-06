import itertools
import logging
import uuid
import numpy as np
from predictor import AccType, Predictor


class Executor:
    def __init__(self, isi):
        self.id = uuid.uuid4().hex
        self.isi = isi
        self.predictors = {}
        self.num_predictor_types = np.zeros(4)
        self.assigned_requests = {}
        self.iterator = itertools.cycle(self.predictors)
        # EITHER: do we want a separate event queue for each executor? then we would need to
        # have another clock and interrupt when request ends
        # OR: a better way would be to just have predictors in the executor that we mark
        # busy when a request comes to them
        # if all predictors are busy, then the request either waits or has to fail
        # 3 ways to do this: round robin, earliest start time, earliest finish time (EFT)

    def add_predictor(self, acc_type=AccType.CPU):
        # print('acc_type: {}'.format(acc_type.value))
        predictor = Predictor(acc_type.value)
        self.predictors[predictor.id] = predictor
        self.num_predictor_types[acc_type.value-1] += 1
        self.iterator = itertools.cycle(self.predictors)
        return id

    
    def remove_predictor_by_id(self, id):
        if id in self.predictors:
            predictor_type = self.predictors[id].acc_type.value
            self.num_predictor_types[predictor_type-1] -= 1
            del self.predictors[id]
            self.iterator = itertools.cycle(self.predictors)
            return True
        else:
            return False

    
    def remove_predictor_by_type(self, acc_type):
        ''' If predictor of given type exists, remove it and return True.
            Otherwise, return False.
        '''
        for id in self.predictors:
            predictor_type = self.predictors[id].acc_type
            if acc_type == predictor_type:
                self.num_predictor_types[predictor_type-1] -= 1
                del self.predictors[id]
                self.iterator = itertools.cycle(self.predictors)
                return True
        return False


    def process_request(self, event, clock, runtimes):
        if len(self.predictors) == 0:
            return None

        # Step 1: load balance
        # for now, we use round robin
        predictor = self.predictors[next(self.iterator)]

        # Step 2: read up runtime based on the predictor type selected
        runtime = runtimes[predictor.acc_type][event.desc]
        event.runtime = runtime
        # print('executor.py -- Request runtime read from dict: {}'.format(runtime))

        # Step 3: assign to predictor selected
        assigned = predictor.assign_request(event, clock)
        if assigned is not None:
            self.assigned_requests[event.id] = predictor
            # print('self.assigned_requests: {}'.format(self.assigned_requests))
        else:
            logging.debug('WARN: Request id {} for {} could not be assigned to any predictor. (Time: {})'.format(event.id, event.desc, clock))

        return assigned

    
    def finish_request(self, event, clock):
        # print('self.assigned_requests: {}'.format(self.assigned_requests))
        if event.id not in self.assigned_requests:
            logger.error('ERROR: Could not find assigned request.')
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
                    