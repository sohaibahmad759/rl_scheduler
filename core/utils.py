import time
import numpy as np


def log_throughput(logger, observation, simulation_time, allocation_window,
                   total_accuracy, total_successful, dropped, late,
                   estimated_throughput, estimated_effective_accuracy,
                   ilp_utilization, proportional_lb, ilp_demand,
                   demand_ewma):
    ''' Writes aggregate throughput and accuracy logs in the following CSV format:
    wallclock_time,simulation_time,demand,throughput,capacity
    '''
    demand = np.sum(observation[:, -2])
    failed_request_rate = np.sum(observation[:, -1])
    throughput = demand - failed_request_rate

    latency_matrix = observation[:-1, 4:8]
    throughput_matrix = 1000 / latency_matrix
    allocation_matrix = observation[:-1, 0:4]

    capacity_matrix = np.multiply(allocation_matrix, throughput_matrix)
    capacity = np.sum(capacity_matrix) * allocation_window / 1000

    if total_successful == 0:
        effective_accuracy = 0
    else:
        effective_accuracy = total_accuracy / total_successful

    # logger.debug('latency matrix:' + str(latency_matrix))
    # logger.debug('allocation matrix:' + str(allocation_matrix))
    # logger.debug('capacity matrix:' + str(capacity_matrix))
    
    logger.info(f'{time.time()},{simulation_time},{demand},{throughput},{capacity},'
                f'{effective_accuracy:.2f},{total_accuracy:.2f},{total_successful},'
                f'{dropped},{late},{estimated_throughput:.1f},'
                f'{estimated_effective_accuracy:.2f},{ilp_utilization:.2f},'
                f'{proportional_lb},{ilp_demand:.2f},{demand_ewma:.2f}')

def log_thput_accuracy_per_model(logger, simulation_time, requests, failed, accuracy):
    ''' Writes throughput and accuracy logs per model in the following CSV format:
    wallclock_time,simulation_time,demand_nth_model,throughput_nth_model,normalized_throughput_nth_model,accuracy_nth_model
    '''
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
