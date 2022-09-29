import time
import numpy as np


def log_throughput(logger, observation, simulation_time, allocation_window):
    demand = np.sum(observation[:, -2])
    failed_request_rate = np.sum(observation[:, -1])
    throughput = demand - failed_request_rate

    latency_matrix = observation[:-1, 4:8]
    throughput_matrix = 1000 / latency_matrix
    allocation_matrix = observation[:-1, 0:4]

    capacity_matrix = np.multiply(allocation_matrix, throughput_matrix)
    capacity = np.sum(capacity_matrix) * allocation_window / 1000

    # logger.debug('latency matrix:' + str(latency_matrix))
    # logger.debug('allocation matrix:' + str(allocation_matrix))
    # logger.debug('capacity matrix:' + str(capacity_matrix))
    
    logger.info(str(time.time()) + ',' + str(simulation_time) + ',' + str(demand) \
                    + ',' + str(throughput) + ',' + str(capacity))

def log_thput_accuracy_per_model(logger, simulation_time, requests, failed, accuracy):
    line = str(time.time()) + ',' + str(simulation_time)
    for model in range(len(requests)):
        throughput = requests[model] - failed[model]
        line = line + ',' + str(requests[model]) + ',' + str(throughput) + ','
        if requests[model] > 0:
            line = line + str(throughput/requests[model])
        else:
            line = line + '0'
        line = line + ',' + str(accuracy[model])
    logger.info(line)
