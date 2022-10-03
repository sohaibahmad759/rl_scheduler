import uuid
from enum import Enum


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
    def __init__(self, acc_type=AccType.CPU, qos_level=0, profiled_accuracy=100.0,
                    profiled_latencies={}, variant_name=None, executor=None,
                    batch_size=1):
        # attributes related to predictor hardware
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
        self.batch_size = batch_size

        self.load = None

        self.executor = executor

    
    def set_load(self, load):
        self.load = load
    

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
        del self.request_dict[event.id]
        if len(self.request_dict) == 0:
            self.busy = False
            self.busy_till = None

        return True

    
    def assign_batch(self, clock):
        # If request can be finished within deadline, return end_time, else return None (failed request)
        temp_queue = []
        dequeued_requests = 0

        while dequeued_requests < self.batch_size or len(self.request_queue) > 0:
            temp_queue.append(self.request_queue.pop(0))

        # TODO: Check profiled latency for a given batch size
        end_time = self.profiled_latencies[]
        while len(temp_queue) > 0:
            current_request = temp_queue.pop(0)

            self.busy = True
            self.busy_till = end_time
            self.request_dict[current_request.id] = 1

        return end_time

        return
    
