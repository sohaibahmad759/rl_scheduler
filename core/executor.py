import itertools
import logging
import math
import random
import time
import uuid
import numpy as np
from types import NoneType
from enum import Enum
from core.predictor import AccType, Predictor


class Behavior(Enum):
    BESTEFFORT = 1
    STRICT = 2


class TaskAssignment(Enum):
    RANDOM = 1
    ROUND_ROBIN = 2
    EARLIEST_FINISH_TIME = 3
    LATEST_FINISH_TIME = 4
    INFAAS = 5
    CANARY = 6


class Executor:
    def __init__(self, isi, task_assignment, n_qos_levels=1, behavior=Behavior.BESTEFFORT, runtimes=None,
                    variant_runtimes=None, variant_loadtimes=None, max_acc_per_type=0, simulator=None):
        self.id = uuid.uuid4().hex
        self.isi = isi
        self.n_qos_levels = n_qos_levels
        self.predictors = {}
        self.num_predictor_types = np.zeros(4 * n_qos_levels)
        self.max_acc_per_type = max_acc_per_type
        self.assigned_requests = {}
        self.iterator = itertools.cycle(self.predictors)
        self.behavior = behavior
        self.task_assignment = TaskAssignment(task_assignment)
        self.runtimes = runtimes

        self.variant_runtimes = variant_runtimes
        self.variant_loadtimes = variant_loadtimes
        self.variant_accuracies = {}

        self.model_variants = {}
        self.variants_for_this_executor = []

        # Mapping from model variant to percentage of requests to send to that variant
        self.canary_routing_table = {}

        self.simulator = simulator
        
        # EITHER: do we want a separate event queue for each executor? then we would need to
        # have another clock and interrupt when request ends
        # OR: a better way would be to just have predictors in the executor that we mark
        # busy when a request comes to them
        # if all predictors are busy, then the request either waits or has to fail
        # 4 ways to do this: round robin, earliest start time, earliest finish time (EFT), latest finish time


    def add_predictor(self, acc_type=AccType.GPU, qos_level=0, variant_name=None):
        # print('acc_type: {}'.format(acc_type.value))
        # profiled_latencies = self.runtimes[acc_type.value]
        profiled_latencies = self.variant_runtimes[acc_type.value]
        
        if variant_name is None:
            min_accuracy = 100.0
            for candidate in self.model_variants[self.isi]:
                candidate_accuracy = self.variant_accuracies[(self.isi, candidate)]
                if candidate_accuracy < min_accuracy:
                    min_accuracy = candidate_accuracy
                    variant_name = candidate
        
        profiled_accuracy = self.variant_accuracies[(self.isi, variant_name)]

        # print('Model variant selected: {}, Profiled accuracy: {}'.format(variant_name, profiled_accuracy))
        # print(self.variant_accuracies)
        # time.sleep(5)

        predictor = Predictor(acc_type.value, qos_level=qos_level, profiled_accuracy=profiled_accuracy,
                                profiled_latencies=profiled_latencies, variant_name=variant_name,
                                executor=self, simulator=self.simulator)
        self.predictors[predictor.id] = predictor
        self.num_predictor_types[acc_type.value-1 + qos_level*4] += 1
        self.iterator = itertools.cycle(self.predictors)

        ## Set an initial weight for canary routing, which will later get updated as the ILP runs
        # if len(self.canary_routing_table) == 0:
        #     self.canary_routing_table[predictor.id] = 1.0
        # else:
        #     self.canary_routing_table[predictor.id] = sum(list(self.canary_routing_table.values()))/len(self.canary_routing_table)
        # print(f'Weight set for predictor {predictor.id}: {self.canary_routing_table[predictor.id]}')

        return predictor.id


    def set_runtimes(self, runtimes=None):
        self.runtimes = runtimes

    
    def set_loadtimes(self, loadtimes=None):
        self.loadtimes = loadtimes

    
    def set_model_variants(self, model_variants={}):
        self.model_variants = model_variants
        self.variants_for_this_executor = model_variants[self.isi]
        self.initialize_routing_table()

    
    def initialize_routing_table(self):
        ''' Initializes the routing table to set equal probabilities for all
        model variants
        '''
        if len(self.model_variants) == 0:
            print(f'initialize_routing_table: no model variants set for executor {self}')
            time.sleep(10)
            return

        routing_table = {}
        model_variants = self.model_variants[self.isi]
        total_variants = len(model_variants)

        for model_variant in model_variants:
            routing_table[model_variant] = 1.0 / total_variants

        self.canary_routing_table = routing_table
        return


    def set_variant_accuracies(self, accuracies=None):
        self.variant_accuracies = accuracies

    
    def set_variant_runtimes(self, runtimes=None):
        self.variant_runtimes = runtimes

    
    def set_variant_loadtimes(self, loadtimes=None):
        self.variant_loadtimes = loadtimes

    
    def remove_predictor_by_id(self, id):
        if id in self.predictors:
            predictor_type = self.predictors[id].acc_type
            predictor_qos = self.predictors[id].qos_level
            self.num_predictor_types[predictor_type-1 + predictor_qos*4] -= 1
            del self.predictors[id]
            # del self.canary_routing_table[id]
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
                # del self.canary_routing_table[id]
                self.iterator = itertools.cycle(self.predictors)
                return True
        return False


    def process_request(self, event, clock, runtimes):
        if len(self.predictors) == 0:
            self.add_predictor()
            # return None, False, 0
            
        qos_met = True

        # Step 1: load balance

        # filter out predictors that match the request's QoS level
        filtered_predictors = list(filter(lambda key: self.predictors[key].qos_level >= event.qos_level, self.predictors))
        
        # TODO: is this the implementation of behavior we want? If there is any matching QoS level,
        #       it will be used regardless of finish time or any other metric
        # TODO: what to do if there are no predictors matching QoS level?
        if len(filtered_predictors) == 0:
            qos_met = False
            logging.debug('No predictors found for QoS level {} for request of {}'.format(event.qos_level, event.desc))
            if self.behavior == Behavior.BESTEFFORT:
                # choose from any QoS level
                filtered_predictors = list(self.predictors.keys())
            # TODO: increment missed QoS counter (histogram)

        if self.task_assignment == TaskAssignment.RANDOM:
            predictor = self.predictors[random.choice(filtered_predictors)]
        elif self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME or self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
            finish_times = []
            for key in filtered_predictors:
                candidate = self.predictors[key]
                potential_runtime = candidate.profiled_latencies[(event.desc, event.qos_level)]
                if candidate.busy:
                    finish_time = candidate.busy_till + potential_runtime
                else:
                    finish_time = clock + potential_runtime

                # finish_time has to be within the request's deadline
                if finish_time > clock + event.deadline:
                    # set it to an invalid value
                    if self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME:
                        finish_time = np.inf
                    elif self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
                        finish_time = -1
                finish_times.append(finish_time)
            logging.debug('filtered_predictors: {}'.format(filtered_predictors))
            logging.debug('finish_times: {}'.format(finish_times))
            if self.task_assignment == TaskAssignment.EARLIEST_FINISH_TIME:
                idx = finish_times.index(min(finish_times))
            elif self.task_assignment == TaskAssignment.LATEST_FINISH_TIME:
                idx = finish_times.index(max(finish_times))
            logging.debug('idx: {}'.format(idx))
            predictor = self.predictors[filtered_predictors[idx]]
            logging.debug('filtered_predictors[idx]: {}'.format(filtered_predictors[idx]))
            logging.debug('predictor: {}'.format(predictor))
            # time.sleep(2)

        elif self.task_assignment == TaskAssignment.CANARY:
            predictor = None

            selected_variant = random.choices(list(self.canary_routing_table.keys()),
                                    weights=list(self.canary_routing_table.values()),
                                    k=1)[0]
            # Canary routing only tells us the model variant to use, but does not
            # tell us which instance of that model variant. We therefore randomly
            # choose different instances of the model variant, with the expectation
            # that with a large enough number of requests, we will have spread out
            # the requests evenly to all instances (law of large numbers)

            # TODO: The peak profiled throughput on different instances of the same
            # model variant hosted on different accelerators will be different.
            # So instead of evenly spreading out the requests, it would make more
            # sense to spread requests proportionally
            variants = list(filter(lambda x: self.predictors[x].variant_name == self.predictors[selected_variant].variant_name,
                                    self.predictors))
            selected_predictor_id = random.choice(variants)
            selected_predictor = self.predictors[selected_predictor_id]
            predictor = selected_predictor

            # selected_predictor.enqueue_request(event, clock)
            # self.assigned_requests[event.id] = selected_predictor

            # First choose model variant

            # Then within that model variant, perform earliest finish time


        elif self.task_assignment == TaskAssignment.INFAAS:
            # TODO: add dynamic batching to INFaaS
            batch_size = 1

            accuracy_filtered_predictors = list(filter(lambda key: self.predictors[key].profiled_accuracy >= event.accuracy, self.predictors))
            print(f'accuracy_filtered_predictors: {accuracy_filtered_predictors},'
                  f' all predictors for this executor {self.isi}: {self.predictors}')
            # time.sleep(1)
            predictor = None
            infaas_candidates = []
            not_found_reason = 'None'
            # There is atleast one predictor that matches the accuracy requirement of the request
            if len(accuracy_filtered_predictors) > 0:
                # If there is any predictor that can meet request
                for key in accuracy_filtered_predictors:
                    _predictor = self.predictors[key]
                    print(f'infaas predictor profiled latencies: {_predictor.profiled_latencies}')
                    time.sleep(10)
                    peak_throughput = math.floor(1000 /  _predictor.profiled_latencies[(event.desc, 
                                                    event.qos_level)])
                    queued_requests = len(_predictor.request_dict)

                    logging.debug(f'Throughput: {peak_throughput}')
                    logging.debug(f'Queued requests: {queued_requests}')
                    if peak_throughput > queued_requests:
                        _predictor.set_load(float(queued_requests)/peak_throughput)
                        infaas_candidates.append(_predictor)
                    else:
                        continue
                
                # We have now found a list of candidate predictors that match both accuracy
                # and deadline of the request, sorted by load
                infaas_candidates.sort(key=lambda x: x.load)

                # for candidate in infaas_candidates:
                #     print('infaas candidate:' + str(candidate) + ', load:' + str(candidate.load))
                # print('infaas candidates:' + str(infaas_candidates))
                # time.sleep(5)

                if len(infaas_candidates) > 0:
                    # Select the least loaded candidate
                    predictor = infaas_candidates[0]
                    logging.debug('found')
                else:
                    # There is no candidate that meets deadline
                    not_found_reason = 'latency_not_met'
                    logging.debug('not found')
            else:
                # There is no predictor that even matches the accuracy requirement of the request
                not_found_reason = 'accuracy_not_met'
                logging.debug('not found')
            
            # No predictor has been found yet, either because accuracy not met or deadline not met
            if predictor is None:
                # Now we try to find an inactive model variant that can meet accuracy+deadline
                isi_name = event.desc
                inactive_candidates = {}
                checked_variants = set(map(lambda key: self.predictors[key].variant_name, self.predictors))
                print(f'checked variants: {checked_variants}')
                # time.sleep(1)

                # print('model variants:' + str(self.model_variants))
                # print()
                # print('model variant accuracies:' + str(self.variant_accuracies))
                # print()
                # print('model variant runtimes:' + str(self.variant_runtimes))
                # print()
                # print('model variant loadtimes:' + str(self.variant_loadtimes))
                # print()
                for model_variant in self.model_variants[isi_name]:
                    # If we already checked for this variant
                    if model_variant in checked_variants:
                        continue
                    else:
                        for acc_type in AccType:
                            predictor_type = acc_type.value
                            # TODO: this should not just use the batch_size variable from before, especially 1
                            runtime = self.variant_runtimes[predictor_type][(isi_name, model_variant, batch_size)]

                            print(f'infaas, runtime: {runtime}, deadline: {event.deadline}')

                            largest_batch_sizes = self.simulator.get_largest_batch_sizes()
                            if acc_type.value == 1:
                                sim_acc_type = 'CPU'
                            elif acc_type.value == 2:
                                sim_acc_type = 'GPU_AMPERE'
                            elif acc_type.value == 3:
                                sim_acc_type = 'VPU'
                            elif acc_type.value == 4:
                                sim_acc_type = 'GPU_PASCAL'
                            largest_batch_size = largest_batch_sizes[(sim_acc_type, model_variant)]

                            if math.isinf(runtime) or largest_batch_size == 0:
                                print(f'largest_batch_size: {largest_batch_size}, runtime: {runtime}')
                                # time.sleep(1)
                                continue
                            
                            loadtime = self.variant_loadtimes[(isi_name, model_variant)]
                            total_time = runtime + loadtime

                            if total_time < event.deadline:
                                inactive_candidates[(model_variant, acc_type)] = total_time
                logging.info(f'inactive candidates: {inactive_candidates}')
                # time.sleep(1)

                for candidate in inactive_candidates:
                    model_variant, acc_type = candidate

                    if self.num_predictor_types[acc_type.value-1 + event.qos_level*4] < self.max_acc_per_type:
                        predictor_id = self.add_predictor(acc_type=acc_type, variant_name=model_variant)
                        predictor = self.predictors[predictor_id]
                        logging.info(f'Predictor {predictor} added from inactive variants')
                        # time.sleep(1)
                        break

            # (Line 8) If we still cannot find one, we try to serve with the closest
            # possible accuracy and/or deadline
            if predictor is None:
                logging.debug('No predictor found from inactive variants either')
                closest_acc_candidates = sorted(self.predictors, key=lambda x: abs(self.predictors[x].profiled_accuracy - event.accuracy))
                # print('event accuracy:' + str(event.accuracy))
                # print('closest accuracy candidates:' + str(closest_acc_candidates))
                for candidate in closest_acc_candidates:
                    _predictor = self.predictors[candidate]
                    variant_name = _predictor.variant_name
                    acc_type = _predictor.acc_type

                    runtime = self.variant_runtimes[acc_type][(isi_name, variant_name, batch_size)]

                    # peak_throughput = math.floor(1000 /  _predictor.profiled_latencies[(event.desc, 
                    #                                 event.qos_level)])
                    peak_throughput = math.floor(1000 /  _predictor.profiled_latencies[(event.desc, 
                                                    _predictor.variant_name, batch_size)])
                    queued_requests = len(_predictor.request_dict)

                    logging.debug('Throughput:', peak_throughput)
                    logging.debug('Queued requests:', queued_requests)
                    if peak_throughput > queued_requests and runtime <= event.deadline:
                        _predictor.set_load(float(queued_requests)/peak_throughput)
                        predictor = _predictor
                        logging.debug('closest predictor {} found'.format(predictor))
                        break
                    else:
                        continue
            
        if predictor is None:
            # No predictor, there is literally nothing that can be done at this point except to drop request
            logging.info('No predictor available whatsoever')
            assigned = None
            qos_met = False
            return assigned, qos_met, 0

        else:
            # Predictor has been found, assign request to it

            # round-robin:
            # predictor = self.predictors[next(self.iterator)]

            # At this point, the variable 'predictor' should indicate which predictor has been
            # selected by one of the heuristics above

            # Step 2: read up runtime based on the predictor type selected
            runtime = self.variant_runtimes[predictor.acc_type][(event.desc, predictor.variant_name, batch_size)]
            # runtime = runtimes[predictor.acc_type][(event.desc, event.qos_level)]
            event.runtime = runtime
            # print('executor.py -- Request runtime read from dict: {}'.format(runtime))

            # Step 3: assign to predictor selected
            assigned = predictor.assign_request(event, clock)
            accuracy = predictor.profiled_accuracy
            if assigned is not None:
                self.assigned_requests[event.id] = predictor
                # print('self.assigned_requests: {}'.format(self.assigned_requests))
            else:
                logging.debug('WARN: Request id {} for {} could not be assigned to any predictor. (Time: {})'.format(event.id, event.desc, clock))

            return assigned, qos_met, accuracy
    

    def trigger_infaas_upscaling(self):
        logging.debug('infaas upscaling triggered')
        batch_size = 1
        for key in self.predictors:
            predictor = self.predictors[key]
            # peak_throughput = 1000 / predictor.profiled_latencies[(self.isi, predictor.qos_level)]
            peak_throughput = 1000 / self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            queued_requests = len(predictor.request_dict)

            infaas_slack = self.simulator.infaas_slack

            if math.floor(peak_throughput * infaas_slack) > queued_requests:
                # no need for scaling
                logging.debug('no need for scaling')
            else:
                logging.info(f'INFaaS autoscaling triggered at predictor {predictor.id}, '
                             f'executor {self.isi}')
                logging.info(f'floor(peak_throughput * infaas_slack): {(math.floor(peak_throughput * infaas_slack))}')
                logging.info(f'queued requests: {queued_requests}')

                # Now we have two options: (1) replication, (2) upgrading to meet SLO
                # Calculate the cost for both and choose the cheaper option

                # Option 1: Replication
                incoming_load = self.simulator.total_requests_arr[self.simulator.isi_to_idx[self.isi]]
                logging.info('incoming load:' + str(incoming_load))
                replicas_needed = math.ceil(incoming_load / (peak_throughput * infaas_slack))

                # Option 2: Upgrade
                # We might not upgrade to a model variant with higher accuracy, but we
                # could upgrade to a different accelerator type with higher throughput
                cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi, predictor.variant_name, batch_size)]
                gpu_peak_throughput = self.variant_runtimes[AccType.GPU.value][(self.isi, predictor.variant_name, batch_size)]
                vpu_peak_throughput = self.variant_runtimes[AccType.VPU.value][(self.isi, predictor.variant_name, batch_size)]
                fpga_peak_throughput = self.variant_runtimes[AccType.FPGA.value][(self.isi, predictor.variant_name, batch_size)]
                
                cpu_needed = math.ceil(incoming_load / (cpu_peak_throughput * infaas_slack))
                gpu_needed = math.ceil(incoming_load / (gpu_peak_throughput * infaas_slack))
                vpu_needed = math.ceil(incoming_load / (vpu_peak_throughput * infaas_slack))
                fpga_needed = math.ceil(incoming_load / (fpga_peak_throughput * infaas_slack))

                cpu_available = self.simulator.available_predictors[0]
                gpu_available = self.simulator.available_predictors[1]
                vpu_available = self.simulator.available_predictors[2]
                fpga_available = self.simulator.available_predictors[3]

                if cpu_available == 0:
                    cpu_needed = math.inf
                if gpu_available == 0:
                    gpu_needed = math.inf
                if vpu_available == 0:
                    vpu_needed = math.inf
                if fpga_available == 0:
                    fpga_needed = math.inf

                # This is where the unit cost is playing in: we are choosing the
                # minimum number of accelerators that can meet requirements,
                # rather than minimizing some non-uniform cost across them

                # If we were trying to minimize cost as dollar values...

                # If we were trying to minimize cost as accuracy drop, we would
                # sort the accelerators in order of increasing accuracy drop,
                # then choose accelerators sequentially from that ordering.
                # Problem: we do not consider throughput in that ordering.
                # Would this problem exist and is tolerated in dollar cost version
                # as well?

                # From INFaaS:
                # Each worker runs a model-autoscaler that approximates ILP as follows:
                # (a) Identify whether the constraints are in
                # danger of being violated,
                
                # (b) Consider two strategies, replicate
                # or upgrade/downgrade, to satisfy the constraints,

                # (c) Compute the objective for each of these scaling actions 
                # and pick the one that minimizes the objective cost function

                # TODO: can we have combinations of different types of accelerators?
                upgrade_needed = min([cpu_needed, gpu_needed, vpu_needed, fpga_needed])

                if upgrade_needed <= replicas_needed:
                    type_needed = [cpu_needed, gpu_needed, vpu_needed, fpga_needed].index(upgrade_needed)

                    logging.info('upgrade needed:' + str(upgrade_needed))
                    logging.info('type needed:' + str(type_needed))
                    logging.info('available predictors left:' +
                                 str(self.simulator.available_predictors[type_needed]))
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and upgrade_needed > 0:
                        self.add_predictor(acc_type=AccType(type_needed+1))
                        upgrade_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                    logging.info('upgrade needed:' + str(upgrade_needed))
                    logging.info('available predictors left:' +
                                 str(self.simulator.available_predictors[type_needed]))
                    # time.sleep(2)
                    return
                else:
                    # add 'replicas_needed' predictors of type 'predictor.acc_type' or as many as possible
                    type_needed = predictor.acc_type
                    logging.info('replicas needed:' + str(replicas_needed))
                    logging.info('type needed:' + str(type_needed))
                    logging.info('available predictors left:' +
                                 str(self.simulator.available_predictors[type_needed]))
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and replicas_needed > 0:
                        self.add_predictor(acc_type=type_needed)
                        replicas_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                    logging.info('replicas needed:' + str(upgrade_needed))
                    logging.info('available predictors left:' +
                                 str(self.simulator.available_predictors[type_needed]))
                    # time.sleep(2)
                    return

    
    def trigger_infaas_v2_upscaling(self):
        logging.info('infaas v2 upscaling triggered')
        # TODO: how to decide the batch size? It is probably decided by the
        #       model autoscaling policy
        # TODO: Check if batch size variable is used in other places by mistake
        for key in self.predictors:
            predictor = self.predictors[key]
            batch_size = predictor.get_infaas_batch_size()
            runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            peak_throughput = batch_size * 1000 / runtime
            queued_requests = len(predictor.request_dict)

            infaas_slack = 0.95

            # TODO: should this be > or >=? previously it was set to be >, but it
            #       gets triggered even if queued requests are 0 and floor(peak throughput * slack)
            #       is 0
            if math.floor(peak_throughput * infaas_slack) >= queued_requests:
                # no need for scaling
                logging.info(f'no need for upscaling, queued_requests: {queued_requests}')
            else:
                logging.info(f'INFaaS autoscaling triggered at predictor {predictor.id}, '
                             f'executor {self.isi}')
                logging.info(f'peak_throughput: {peak_throughput}')
                logging.info(f'floor(peak_throughput * infaas_slack): {(math.floor(peak_throughput * infaas_slack))}')
                logging.info(f'queued requests: {queued_requests}')

                # First we consider batch size increase, which costs 0 to change
                # Technically it is an upgrade, but we can increase it first and
                # see if anything still needs to be changed
                for batch_size in self.simulator.allowed_batch_sizes:
                    if batch_size > predictor.get_largest_batch_size():
                        break
                    else:
                        predictor.set_infaas_batch_size(batch_size)
                
                peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime

                # Now we have two options: (1) replication, (2) upgrading to meet SLO
                # Calculate the cost for both and choose the cheaper option

                # Option 1: Replication
                incoming_load = self.simulator.total_requests_arr[self.simulator.isi_to_idx[self.isi]]
                logging.info(f'incoming load: {incoming_load}')
                replicas_needed = math.ceil(incoming_load / (peak_throughput * infaas_slack))

                # In INFaaS v2, to calculate the cost of replicas_needed, we also
                # need to know what type of replicas are needed, since the cost
                # i.e. accuracy drop, depends on the model variant (not accelerator)

                # Solution:
                # well actually that makes it quite simple. If we don't need to know
                # accelerators and just need model variant to calculate cost, replicating
                # this given predictor will have as much cost as the cost of this predictor
                # i.e., the accuracy drop of this predictor

                logging.info(f'replicas needed: {replicas_needed}, accuracy drop of '
                             f'this variant: {predictor.get_infaas_cost()}')
                # The cost for all replicas of this predictor is the same
                cost_replication = predictor.get_infaas_cost() * replicas_needed
                logging.info(f'total cost of replication: {cost_replication}')

                # Option 2: Upgrade
                # Old: We might not upgrade to a model variant with higher accuracy, but we
                # could upgrade to a different accelerator type with higher throughput
                # NEW: Upgrade could also mean upgrading to a higher batch size,
                #      or upgrading to a model variant with different/higher accuracy

                # Option 2a: Change to a different model variant

                # Consider the different model variants
                # TODO: How do we compare the new peak throughput of a single variant change (2a)
                # with the peak throughput of multiple potential predictors (2b)?
                # Based on Section 4.2.2, "Scaling up algorithm", it seems we have to
                # run (2b) for every model variant of this model family that can meet
                # SLO and support a higher throughput than the currently running variant
                # We calculate the cost_upgrade for each of these variants, then take
                # the minimum cost_upgrade and compare that with cost_replication, to take
                # the appropriate action
                model_variants = dict(filter(lambda x: x[0][0]==self.isi, predictor.profiled_latencies.items()))
                print(f'self.isi: {self.isi}, model_variants: {model_variants}')
                cost_upgrade = np.inf
                upgrade_needed = None
                type_needed = None
                selected_variant = None
                for model_variant in model_variants:
                    variant_name = model_variant[1]
                    batch_size = model_variant[2]
                    if batch_size > predictor.get_largest_batch_size():
                        continue

                    cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi, variant_name, batch_size)]
                    gpu_peak_throughput = self.variant_runtimes[AccType.GPU.value][(self.isi, variant_name, batch_size)]
                    vpu_peak_throughput = self.variant_runtimes[AccType.VPU.value][(self.isi, variant_name, batch_size)]
                    fpga_peak_throughput = self.variant_runtimes[AccType.FPGA.value][(self.isi, variant_name, batch_size)]
                    
                    cpu_needed = math.ceil(incoming_load / (cpu_peak_throughput * infaas_slack))
                    gpu_needed = math.ceil(incoming_load / (gpu_peak_throughput * infaas_slack))
                    vpu_needed = math.ceil(incoming_load / (vpu_peak_throughput * infaas_slack))
                    fpga_needed = math.ceil(incoming_load / (fpga_peak_throughput * infaas_slack))

                    cpu_available = self.simulator.available_predictors[0]
                    gpu_available = self.simulator.available_predictors[1]
                    vpu_available = self.simulator.available_predictors[2]
                    fpga_available = self.simulator.available_predictors[3]

                    if cpu_available == 0 or self.get_largest_batch_size(variant_name, 1) == 0:
                        cpu_needed = math.inf
                    if gpu_available == 0 or self.get_largest_batch_size(variant_name, 2) == 0:
                        gpu_needed = math.inf
                    if vpu_available == 0 or self.get_largest_batch_size(variant_name, 3) == 0:
                        vpu_needed = math.inf
                    if fpga_available == 0 or self.get_largest_batch_size(variant_name, 4) == 0:
                        fpga_needed = math.inf

                    upgrade_needed_variant = min([cpu_needed, gpu_needed, vpu_needed, fpga_needed])
                    type_needed_variant = [cpu_needed, gpu_needed, vpu_needed, fpga_needed].index(upgrade_needed_variant)
                    cost_upgrade_variant = self.model_variant_infaas_cost(variant_name) * upgrade_needed_variant
                    logging.info(f'variant: {model_variant}, upgrade_needed_variant: {upgrade_needed_variant},' 
                                 f' type_needed_variant: {type_needed_variant}, cost per instance: '
                                 f'{self.model_variant_infaas_cost(variant_name)}')

                    if cost_upgrade_variant < cost_upgrade:
                        cost_upgrade = cost_upgrade_variant
                        upgrade_needed = upgrade_needed_variant
                        type_needed = type_needed_variant
                        selected_variant = variant_name
                print(f'selected variant for upgrade: {selected_variant}, cost '
                      f'of upgrade: {cost_upgrade}')
                # time.sleep(1)

                # Option 2b: Change to a different hardware accelerator and make multiple copies
                # batch_size = predictor.get_infaas_batch_size()
                # cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi, predictor.variant_name, batch_size)]
                # gpu_peak_throughput = self.variant_runtimes[AccType.GPU.value][(self.isi, predictor.variant_name, batch_size)]
                # vpu_peak_throughput = self.variant_runtimes[AccType.VPU.value][(self.isi, predictor.variant_name, batch_size)]
                # fpga_peak_throughput = self.variant_runtimes[AccType.FPGA.value][(self.isi, predictor.variant_name, batch_size)]
                
                # cpu_needed = math.ceil(incoming_load / (cpu_peak_throughput * infaas_slack))
                # gpu_needed = math.ceil(incoming_load / (gpu_peak_throughput * infaas_slack))
                # vpu_needed = math.ceil(incoming_load / (vpu_peak_throughput * infaas_slack))
                # fpga_needed = math.ceil(incoming_load / (fpga_peak_throughput * infaas_slack))

                # cpu_available = self.simulator.available_predictors[0]
                # gpu_available = self.simulator.available_predictors[1]
                # vpu_available = self.simulator.available_predictors[2]
                # fpga_available = self.simulator.available_predictors[3]

                # if cpu_available == 0:
                #     cpu_needed = math.inf
                # if gpu_available == 0:
                #     gpu_needed = math.inf
                # if vpu_available == 0:
                #     vpu_needed = math.inf
                # if fpga_available == 0:
                #     fpga_needed = math.inf

                # # This is where the unit cost is playing in: we are choosing the
                # # minimum number of accelerators that can meet requirements,
                # # rather than minimizing some non-uniform cost across them

                # # If we were trying to minimize cost as dollar values...

                # # If we were trying to minimize cost as accuracy drop, we would
                # # sort the accelerators in order of increasing accuracy drop,
                # # then choose accelerators sequentially from that ordering.
                # # Problem: we do not consider throughput in that ordering.
                # # Would this problem exist and is tolerated in dollar cost version
                # # as well?

                # # From INFaaS:
                # # Each worker runs a model-autoscaler that approximates ILP as follows:
                # # (a) Identify whether the constraints are in
                # # danger of being violated,
                
                # # (b) Consider two strategies, replicate
                # # or upgrade/downgrade, to satisfy the constraints,

                # # (c) Compute the objective for each of these scaling actions 
                # # and pick the one that minimizes the objective cost function

                # # By this point, we are assuming that same variants need to be added
                
                # # TODO: can we have combinations of different types of accelerators?
                # upgrade_needed = min([cpu_needed, gpu_needed, vpu_needed, fpga_needed])
                # type_needed = [cpu_needed, gpu_needed, vpu_needed, fpga_needed].index(upgrade_needed)

                logging.info(f'upgrade of type {type_needed} needed: {upgrade_needed},'
                             f'accuracy drop of this variant: {predictor.get_infaas_cost()}')
                # cost_upgrade = predictor.get_infaas_cost() * upgrade_needed
                logging.info(f'total cost of upgrade: {cost_upgrade}')
                # time.sleep(1)

                if cost_upgrade <= cost_replication:
                    logging.info(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and upgrade_needed > 0:
                        self.add_predictor(acc_type=AccType(type_needed+1), variant_name=selected_variant)
                        upgrade_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                    logging.info(f'upgrade needed: {upgrade_needed}')
                    logging.info(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # time.sleep(2)
                    return
                else:
                    # add 'replicas_needed' predictors of type 'predictor.acc_type' or as many as possible
                    type_needed = predictor.acc_type - 1
                    logging.info(f'replicas needed: {replicas_needed}')
                    logging.info(f'type needed: {type_needed}')
                    logging.info(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # add 'upgrade_needed' predictors of type 'type_needed' or as many as possible
                    while self.simulator.available_predictors[type_needed] > 0 and replicas_needed > 0:
                        self.add_predictor(acc_type=type_needed)
                        replicas_needed -= 1
                        self.simulator.available_predictors[type_needed] -= 1
                    logging.info(f'replicas needed: {upgrade_needed}')
                    logging.info(f'available predictors left: {self.simulator.available_predictors[type_needed]}')
                    # time.sleep(2)
                    return
        # time.sleep(10)

    
    def trigger_infaas_downscaling(self):
        logging.info('infaas downscaling not implemented')
        # TODO: Each worker determines if the incoming query load can be supported
        #       by removing this instance or downgrading to a cheaper variant (lower
        #       batch size or running on different hardware, perhaps in v2 consider
        #       cost as well)
        # time.sleep(2)
    

    def trigger_infaas_v2_downscaling(self):
        logging.info('infaas v2 downscaling triggered')
        infaas_slack = self.simulator.infaas_slack

        # First find the total load and the capacity of the system
        total_peak_throughput = 0
        total_queued_requests = 0
        for key in self.predictors:
            predictor = self.predictors[key]
            batch_size = predictor.get_infaas_batch_size()
            runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            
            predictor_peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime
            total_peak_throughput += predictor_peak_throughput

            queued_requests = len(predictor.request_dict)
            total_queued_requests += queued_requests
        
        # Now see if any predictors can be removed or downgraded
        predictors_to_remove = []
        predictors_to_add = []
        for key in self.predictors:
            predictor = self.predictors[key]
            runtime = self.variant_runtimes[predictor.acc_type][(self.isi, predictor.variant_name, batch_size)]
            
            predictor_peak_throughput = predictor.get_infaas_batch_size() * 1000 / runtime

            # Two options, (i) removing, (ii) downgrading

            # (i) Removing: If load can be met after removing this predictor
            new_throughput = total_peak_throughput - predictor_peak_throughput
            if new_throughput * infaas_slack > total_queued_requests:
                # remove this predictor
                # self.remove_predictor_by_id(predictor.id)
                predictors_to_remove.append(predictor.id)
                total_peak_throughput = new_throughput
                continue

            # (ii) Downgrading: Consider downgrade options
            downgraded = False

            # We could downgrade to smaller batch size, or running on different hardware
            
            # First consider smaller batch sizes, going in increasing order
            for batch_size in self.simulator.allowed_batch_sizes:
                if batch_size > predictor.get_largest_batch_size():
                    break

                batch_peak_throughput = batch_size * 1000 / runtime
                new_throughput = total_peak_throughput - predictor_peak_throughput + batch_peak_throughput
                if new_throughput * infaas_slack > total_queued_requests:
                    predictor.set_infaas_batch_size(batch_size)
                    total_peak_throughput = new_throughput
                    downgraded = True
                    break
            
            # If we have already downgraded, no need to consider second option
            # for downgrading
            if downgraded:
                continue

            # Now consider running on different hardware
            if predictor.acc_type == AccType.CPU or predictor.acc_type == AccType.VPU:
                # It cannot be downgraded further as CPU/VPU are the slowest
                continue
            else:
                # We are running on GPU or FPGA (in our setting, FPGA is also GPU)
                # If CPU or VPU is available, and load can be supported through that,
                # downgrade
                cpu_available = self.simulator.available_predictors[0]
                vpu_available = self.simulator.available_predictors[2]

                if self.get_largest_batch_size(predictor.variant_name, 'CPU') == 0:
                    continue

                if cpu_available > 0 or vpu_available > 0:
                    cpu_peak_throughput = self.variant_runtimes[AccType.CPU.value][(self.isi,
                                                predictor.variant_name,
                                                predictor.get_infaas_batch_size())]
                    new_throughput = total_peak_throughput - predictor_peak_throughput + cpu_peak_throughput

                    if new_throughput * infaas_slack > total_queued_requests:
                        print(f'removing gpu/fpga predictor and adding cpu/vpu predictor'
                              f' for {predictor.executor.isi}')
                        time.sleep(1)
                        total_peak_throughput = new_throughput
                        # remove this predictor
                        # self.remove_predictor_by_id(predictor.id)
                        predictors_to_remove.append(predictor.id)
                        # add a CPU or VPU predictor, whichever is available
                        if cpu_available > 0:
                            predictors_to_add.append((AccType.CPU, predictor.variant_name))
                            # self.add_predictor(acc_type=AccType.CPU, variant_name=predictor.variant_name)
                        elif vpu_available > 0:
                            predictors_to_add.append((AccType.VPU, predictor.variant_name))
                            # self.add_predictor(acc_type=AccType.VPU, variant_name=predictor.variant_name)
                        else:
                            logging.error('infaas_v2_downscaling: neither CPU nor VPU is available!')
                            time.sleep(10)
        
        # We can't add/remove predictors inside the above loop since it will
        # change the size of the loop
        for id in predictors_to_remove:
            self.remove_predictor_by_id(id)

        for tuple in predictors_to_add:
            acc_type, variant_name = tuple
            self.add_predictor(acc_type=acc_type, variant_name=variant_name)

        return

    
    def model_variant_infaas_cost(self, model_variant):
        ''' Returns the accuracy cost of a model variant
        '''
        isi = self.simulator.get_isi_from_variant_name(model_variant)
        logging.debug(f'model_variant_infaas_cost, isi: {isi}, model_variant: '
                      f'{model_variant}, self.variant_accuracies: {self.variant_accuracies}')
        model_variants = dict(filter(lambda x: x[0][0]==isi, self.variant_accuracies.items()))
        logging.debug(f'model_variants: {model_variants}')
        max_accuracy = max(model_variants.values())
        logging.debug(f'max accuracy: {max_accuracy}')
        cost = max_accuracy - model_variants[(isi, model_variant)]
        logging.debug(f'cost: {cost}')
        return cost

    
    def finish_request(self, event, clock):
        # print('self.assigned_requests: {}'.format(self.assigned_requests))
        if event.id not in self.assigned_requests:
            logging.error('ERROR: Could not find assigned request.')
            time.sleep(10)
            return False
        
        predictor = self.assigned_requests[event.id]

        # TODO: If predictor is removed before request is processed,
        #       count it as a failed request
        # if predictor.id not in self.predictors:
        #     print(f'Predictor was deleted before request could complete')
        #     # time.sleep(10)
        #     return False
        finished = predictor.finish_request(event)
        if finished:
            del self.assigned_requests[event.id]
            logging.debug('Finished request for {} at predictor {}. (Time: {})'.format(event.desc, predictor.id, clock))
            return True
        else:
            logging.debug('WARN: Could not finish request at predictor {}, \
                    executor {}. (Time: {})'.format(predictor.id, self.id, clock))
            time.sleep(5)
            return False

    
    def enqueue_request(self, event, clock):
        ''' When batching is enabled, we send an incoming request to the queue of an
        appropriate predictor. The predictor will dequeue the request and process it
        with a batch of requests.
        '''
        if len(self.predictors) == 0:
            self.add_predictor()

        if self.task_assignment == TaskAssignment.CANARY:
            logging.debug(f'Canary routing table, keys: {list(self.canary_routing_table.keys())}, '
                  f'weights: {list(self.canary_routing_table.values())}, isi: {self.isi}')

            selected_variant = random.choices(list(self.canary_routing_table.keys()),
                                    weights=list(self.canary_routing_table.values()),
                                    k=1)[0]
            logging.debug(f'Selected variant: {selected_variant}')
            
            # Canary routing only tells us the model variant to use, but does not
            # tell us which instance of that model variant. We therefore randomly
            # choose different instances of the model variant, with the expectation
            # that with a large enough number of requests, we will have spread out
            # the requests evenly to all instances (law of large numbers)

            # TODO: The peak profiled throughput on different instances of the same
            # model variant hosted on different accelerators will be different.
            # So instead of evenly spreading out the requests, it would make more
            # sense to spread requests proportionally

            logging.debug(f'self.predictors: {self.predictors}')
            # variants = list(filter(lambda x: x.variant_name == selected_variant, self.predictors))
            # variants = list(filter(lambda x: self.predictors[x].variant_name == self.predictors[selected_variant].variant_name,
            #                         self.predictors))
            variants_dict = dict(filter(lambda x: x[1].variant_name == selected_variant,
                                        self.predictors.items()))
            variants = list(variants_dict.keys())

            if len(variants) == 0:
                self.add_predictor(variant_name=selected_variant)
                variants_dict = dict(filter(lambda x: x[1].variant_name == selected_variant,
                                            self.predictors.items()))
                variants = list(variants_dict.keys())

            logging.debug(f'Variants: {variants}')
            selected_predictor_id = random.choice(variants)
            logging.debug(f'Selected predictor id: {selected_predictor_id}')
            selected_predictor = self.predictors[selected_predictor_id]
            logging.debug(f'Selected predictor: {selected_predictor}')
            # time.sleep(10)

            selected_predictor.enqueue_request(event, clock)
            self.assigned_requests[event.id] = selected_predictor
            return

        else:
            logging.error('Only canary routing is supported with batch request processing')
            exit(0)

    
    def apply_routing_table(self, routing_table={}):
        ''' Applies a given routing table to change the executor's current
        canary routing table
        '''
        logging.debug(f'Executor: {self.isi}, Previous routing table: {self.canary_routing_table},'
                      f' new routing table: {routing_table}')
        if len(routing_table) == 0:
            # TODO: this should not be happening if there are still predictors for this isi
            print(f'Empty routing table passed for isi {self.isi}')
            # time.sleep(10)
            return

        self.canary_routing_table = routing_table
        return

    
    def get_largest_batch_size(self, model_variant, acc_type):
        largest_batch_sizes = self.simulator.get_largest_batch_sizes()

        if acc_type == 1:
            acc_type = 'CPU'
        elif acc_type == 2:
            acc_type = 'GPU_AMPERE'
        elif acc_type == 3:
            acc_type = 'VPU'
        elif acc_type == 4:
            acc_type = 'GPU_PASCAL'
        
        largest_batch_size = largest_batch_sizes[(acc_type, model_variant)]
        return largest_batch_size

    
    def predictors_by_variant_name(self):
        ''' Returns the list of predictors currently hosted by the executor,
        listing the variant name for each predictor
        '''
        predictor_dict = list(map(lambda x: x[1].variant_name, self.predictors.items()))
        return predictor_dict
                    