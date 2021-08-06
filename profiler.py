from typing import Dict, Enum
from pydantic import BaseModel
from microservices.blis_common_models.inference_service import (
    NodeModelType, NodeModelAcceleratorType
)


class Profiler:
    def __init__(self):
        return None


class TrialState(str, Enum):
    """State of the Work item"""
    PENDING = 'pending'
    COMPLETED = 'completed'
    FAILED = 'failed'
    RUNNING = 'running'


class ModelTrialResultsSummary(BaseModel):
    """Results from a Trial of a Model"""
    model_name: str
    trial_id: str
    accelerator: NodeModelAcceleratorType = None
    accelerator_count: int = 1
    state: TrialState = None
    batch_size: int = 1
    latency: float = None # milliseconds
    throughput: float = None # transactions per second
    accuracy: float = None # percentage
    memory_kb: float = None
    power: float = None

 
class ModelTrialResultsDetails(ModelTrialResultsSummary):
    """Additional details about trial for scheduling"""
    predictor_start_time: float = None # milliseconds
 

class ModelIdentification(BaseModel):
    """Information identifying a model in a family"""
    name: str # Must be unique within a model family
    model_type: NodeModelType = NodeModelType.ONNX
    model_md5: str = None # identity hash of model
 

class ModelProfileInfo(ModelIdentification):
    """Information about each model in family"""
    # profile_by_accelerator keyed by Acclerator type, value is a
    # dictionary keyed by batch-size with a value of profilings results
    profile_by_accelerator: Dict[NodeModelAcceleratorType, Dict[int, ModelTrialResultsDetails]]


class ModelFamily(BaseModel):
    """Used to describe a model family that can be used in an ISI"""
    family_id: str
    session_id: str = None # Profiling session associated with family
    members: Dict[str, ModelProfileInfo] # key model_md5
