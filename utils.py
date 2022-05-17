import time
import numpy as np


def log_throughput(logger, observation, simulation_time):
    demand = np.sum(observation[:, -2])
    failed_request_rate = np.sum(observation[:, -1])
    throughput = demand - failed_request_rate
    
    logger.info(str(time.time()) + ',' + str(simulation_time) + ',' + str(demand) \
                    + ',' + str(throughput))