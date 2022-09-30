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
    
    logger.info(f'{time.time()},{simulation_time},{demand},{throughput},{capacity}')

def log_thput_accuracy_per_model(logger, simulation_time, requests, failed, accuracy):
    line = f'{time.time()},{simulation_time}'

    for model in range(len(requests)):
        throughput = requests[model] - failed[model]

        line += f',{requests[model]},{throughput},'
        
        if requests[model] > 0:
            normalized_throughput = throughput/requests[model]
        else:
            normalized_throughput = 0

        line += f'{normalized_throughput},{accuracy[model]}'

    logger.info(line)
