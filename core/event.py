from enum import Enum
import uuid


class EventType(Enum):
    START_REQUEST = 1
    SCHEDULING = 2
    END_REQUEST = 3


class Event:
    def __init__(self, start_time, type, desc, runtime=None, deadline=1000, id='', qos_level=0, accuracy=100.0):
        self.id = id
        if self.id == '':
            self.id = uuid.uuid4().hex
        self.type = type
        self.start_time = start_time
        self.desc = desc
        self.runtime = runtime
        self.deadline = deadline
        self.qos_level = qos_level
        self.accuracy = accuracy
