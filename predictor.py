import uuid
from enum import Enum


class AccType(Enum):
    CPU = 1
    GPU = 2
    VPU = 3
    FPGA = 4


class Predictor:
    def __init__(self, acc_type=AccType.CPU):
        # attributes related to predictor hardware
        self.id = uuid.uuid4().hex
        self.acc_type = acc_type

        # attributes related to current status
        self.busy = False
        self.busy_till = None
        self.request_queue = {}
    

    def assign_request(self, event, clock):
        # if request can be finished within deadline, return end_time, else return None (failed request)
        
        # TODO: request runtime should not be fixed, it should be based on profiled data
        if self.busy:
            end_time = self.busy_till + event.runtime
        else:
            end_time = clock + event.runtime
        if end_time <= event.start_time + event.deadline:
            self.busy = True
            self.busy_till = end_time
            self.request_queue[event.id] = 1
            # self.request_queue.append(event.id)
            return end_time
        else:
            return None

    
    def finish_request(self, event):
        if event.id not in self.request_queue:
            return False
        
        # self.request_queue.remove(event.id)
        del self.request_queue[event.id]
        if len(self.request_queue) == 0:
            self.busy = False
            self.busy_till = None

        return True
    
