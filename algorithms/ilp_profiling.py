import time
import logging
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from algorithms.base import SchedulingAlgorithm
from core.exceptions import IlpException


MINIMUM_BETA = 0.2
MINIMUM_ACCURACY = 0.0
LATENCY_GAP_FACTOR = 1.0


class Ilp(SchedulingAlgorithm):
    def __init__(self, allocation_window, beta, logging_level, starting_allocation=None,
                 static=None, profiling_mode=False):
        SchedulingAlgorithm.__init__(self, 'ILP')

        self.log = logging.getLogger(__name__)
        self.log.addHandler(logging.FileHandler('logs/ilp/output.log', mode='w'))
        self.log.setLevel(logging_level)

        self.allocation_window = allocation_window

        self.initial_beta = beta
        self.beta = beta

        self.simulator = None
        self.num_isi = 0
        
        # We cache the latest solution from run()
        self.cached_solution = None

        self.starting_allocation = starting_allocation
        self.static = static

        self.profiling_mode = profiling_mode

    def is_simulator_set(self):
        if self.simulator is None:
            return False
        else:
            return True

    def set_simulator(self, simulator):
        self.simulator = simulator

        if self.starting_allocation is not None:
            predictor_dict, canary_dict, ilp_x = self.get_solution_from_file(self.starting_allocation)
            self.log.info(f'ilp_x: {ilp_x}')
            self.log.info(f'canary_dict: {canary_dict}')
            self.log.info(f'predictor_dict: {predictor_dict}')
            self.simulator.apply_ilp_solution(predictor_dict, canary_dict, ilp_x)

    def get_solution_from_file(self, file):
        with open(file, mode='r') as rf:
            lines = rf.readlines()
            required_predictors = eval(lines[1].rstrip('\n'))
            canary_dict = eval(lines[3].rstrip('\n'))
            ilp_x = None
            if len(lines) > 4:
                ilp_x = eval(lines[5].rstrip('\n'))

            return required_predictors, canary_dict, ilp_x

    def set_largest_batch_sizes(self, slo_dict, all_model_variants, accelerators):
        ''' Accesses the profiled data from the simulator to establish the maximum
        batch size that does not violate latency SLO for each (variant, accelerator)
        pair
        '''
        max_batch_size_dict = {}
        profiled_latencies = self.simulator.model_variant_runtimes
        
        for isi_name in all_model_variants:
            model_variants = all_model_variants[isi_name]
            for model_variant in model_variants:
                for acc_type in accelerators:
                    acc_latencies = {}
                    if acc_type == 'CPU':
                        acc_latencies = profiled_latencies[1]
                    elif acc_type == 'GPU_AMPERE':
                        acc_latencies = profiled_latencies[2]
                    elif acc_type == 'VPU':
                        acc_latencies = profiled_latencies[3]
                    elif acc_type == 'GPU_PASCAL':
                        acc_latencies = profiled_latencies[4]

                    max_batch_size = 0
                    for batch_size in [1, 2, 4, 8, 16]:
                        latency = np.random.random()*1000
                    # for batch_size in self.simulator.allowed_batch_sizes:
                    #     latency = acc_latencies[(isi_name, model_variant, batch_size)]

                        if batch_size > max_batch_size and latency < slo_dict[isi_name]:
                            max_batch_size = batch_size

                    max_batch_size_dict[(acc_type, model_variant)] = max_batch_size
                    self.log.debug(f'({acc_type}, {model_variant}): {max_batch_size}')
        self.log.debug(f'len(max_batch_size_dict): {len(max_batch_size_dict)}')
        return max_batch_size_dict

    def add_spec_acc_constraints(self, model, x, accelerators, required_predictors):
        self.log.debug(f'model: {model}')
        self.log.debug(f'accelerators: {accelerators}')

        indices = {}
        constraints_added = 0

        for key in required_predictors:
            variant, accelerator_type = key
            instances = required_predictors[key]

            model_family = self.simulator.get_isi_from_variant_name(variant)
            model_variants = set(self.simulator.model_variants[model_family])
            all_variants = set(sum(self.simulator.model_variants.values(), []))

            disallowed_variants = all_variants.difference(model_variants)

            for inst in range(instances):
                idx = indices.get(accelerator_type, 0)

                for disallowed_variant in disallowed_variants:
                    accelerator = f'{accelerator_type}-{idx}'
                    model.addConstr(x[accelerator, disallowed_variant] == 0, f'c_zero_{accelerator}_{disallowed_variant}')
                    constraints_added += 1

                idx += 1
                indices[accelerator_type] = idx

        return model
    
    def disallow_infeasible_variants(self, gurobiModel, x, accelerators, model_variants,
                                     largest_batch_sizes):
        ''' Go through all combinations of model variants and accelerators
        and disallow pairs for which SLO cannot be met (i.e., largest batch size)
        is set to 0
        '''
        for model_variant in model_variants:
            for accelerator in accelerators:
                acc_type = accelerator.split('-')[0]
                
                if largest_batch_sizes[(acc_type, model_variant)] == 0:
                    gurobiModel.addConstr(x[(accelerator, model_variant)] == 0,
                                          f'c_zero_{accelerator}_{model_variant}')
                    
        return gurobiModel

    def run(self, observation, num_acc_types, num_max_acc, num_isi=9,
            variants_per_model=6):
        # num_isi = observation.shape[0] - 1
        # self.num_isi = num_isi
        
        # current_alloc = observation[0:num_isi, 0:num_acc_types]

        # Old way to get demand
        # demand_since_last = observation[0:num_isi, -2]

        self.num_isi = num_isi

        # New way to get demand (EWMA over sliding window)
        # demand_since_last = self.simulator.ewma_demand.ravel()
        # divide demand by time elapsed since last measurement to get demand in units of requests per second
        # demand = demand_since_last / (self.allocation_window / 1000)
        # self.log.info(f'demand: {sum(demand)}')
        # missed_requests = observation[0:num_isi, -1]

        allowed_batch_sizes = [1, 2, 4, 8, 16]

        # models = []
        models = []
        model_families = []
        idx_to_executor = {}
        for dummy_model in range(num_isi):
            dummy_model_name = f'bert_{dummy_model}'
            model_families.append(dummy_model_name)
            idx_to_executor[dummy_model] = dummy_model_name

        # print(f'idx_to_executor: {idx_to_executor}')

        sim_model_variants = {}
        largest_batch_sizes = {}
        variant_counter = 0
        for model_family in model_families:
            sim_model_variants[model_family] = []
            for dummy_variant in range(variants_per_model):
                variant_name = f'variant_{variant_counter}'
                sim_model_variants[model_family].append(variant_name)
                largest_batch_sizes[(model_family, variant_name)] = 1
                variant_counter += 1
            # sim_model_variants[model] = ['variant1'] * variants_per_model

        # print(f'sim_model_variants: {sim_model_variants}')

        # latencies = self.simulator.model_variant_runtimes
        # TODO: have to set this for profiling
        latencies = {}
        for dummy_acc in range(4):
            latencies[dummy_acc+1] = {}
            for model_family in model_families:
                for variants in sim_model_variants[model_family]:
                    for batch_size in allowed_batch_sizes:
                        latencies[dummy_acc+1][(model_family, variants, batch_size)] = np.random.random() * 100

        # print(f'latencies: {latencies}')

        # First generate the parameter variables from the isi_dict
        rtypes = []

        accelerators = []
        for acc in range(num_max_acc):
            accelerators.append('CPU-' + str(acc))
            accelerators.append('GPU_AMPERE-' + str(acc))
            accelerators.append('VPU-' + str(acc))
            accelerators.append('GPU_PASCAL-' + str(acc))

        self.accelerator_dict = {'CPU': 0, 'GPU_AMPERE': 1, 'VPU': 2, 'GPU_PASCAL': 3}

        # A(j)
        # Accuracy of model variant j
        A = {}

        # s(k)
        # Incoming demand (requests per second) for request type k
        s = {}

        # p(i,j)
        # Profiled throughput of model variant j on accelerator i
        p = {}

        # b(j,k)
        # Whether request type k can be served by model variant j
        b = {}

        # This is just for giving shape to z variable
        ik_dict = {}
        
        # largest_batch_sizes = self.simulator.get_largest_batch_sizes()
        # TODO: have to set this for profiling
        # largest_batch_sizes = {('bert', 'variant1'): 1, ('bert', 'variant2'): 1}


        for isi in range(num_isi):
            rtypes.append(isi)

            # Initialize demand for each request type (ISI)
            # s[isi] = np.random.random()*10
            s[isi] = 10

            # isi_name = self.simulator.idx_to_executor[isi]
            # model_variants = self.simulator.model_variants[isi_name]
            isi_name = idx_to_executor[isi]
            model_variants = sim_model_variants[isi_name]

            for model_variant in model_variants:
                models.append(model_variant)

                # A[model_variant] = self.simulator.model_variant_accuracies[(
                #     isi_name, model_variant)]
                A[model_variant] = np.random.random() * 100

                for accelerator in accelerators:
                    accelerator_type = accelerator.split('-')[0]
                    acc_latencies = {}
                    if accelerator_type == 'CPU':
                        acc_latencies = latencies[1]
                    elif accelerator_type == 'GPU_AMPERE':
                        acc_latencies = latencies[2]
                    elif accelerator_type == 'VPU':
                        acc_latencies = latencies[3]
                    elif accelerator_type == 'GPU_PASCAL':
                        acc_latencies = latencies[4]
                    
                    # largest_batch_size = largest_batch_sizes[(accelerator_type, model_variant)]
                    largest_batch_size = 16
                    if largest_batch_size == 0:
                        latency = None
                    else:
                        # latency = acc_latencies[(isi_name, model_variant, largest_batch_size)] * LATENCY_GAP_FACTOR
                        latency = np.random.random()*10

                    if latency is None:
                        throughput = 0
                    else:
                        throughput = largest_batch_size * 1000 / latency
                    p[accelerator, model_variant] = math.floor(throughput)

                    ik_dict[accelerator, isi] = 1

                for isi in range(num_isi):
                    if model_variant in sim_model_variants[idx_to_executor[isi]]:
                        b[model_variant, isi] = 1
                    else:
                        b[model_variant, isi] = 0

        # print(f'b: {b}')
        # print(f'models: {models}')

        # print(f'p: {p}')
        # Helper variables for setting up the ILP
        ij_pairs, _ = gp.multidict(p)
        jk_pairs, _ = gp.multidict(b)
        ik_pairs, _ = gp.multidict(ik_dict)

        solution_found = False
        while solution_found == False and self.beta > MINIMUM_BETA:
            # Set up the optimization model
            m = gp.Model('Accuracy Throughput MILP')
            m.setParam("LogToConsole", 0)

            m.setParam('NonConvex', 2)
            # m.setParam('TimeLimit', 200)
            m.setParam('TimeLimit', 2000)
            m.setParam('MIPGap', 0.5)
            m.setParam('Threads', 12)

            # Add optimization variables
            w = m.addVars(rtypes, name='x')
            x = m.addVars(ij_pairs, vtype=GRB.BINARY, name='x')
            y = m.addVars(accelerators, name='y')
            yp = m.addVars(accelerators, name='yp')
            ypp = m.addVars(accelerators, name='ypp')
            z = m.addVars(ik_pairs, name='z')
            zp = m.addVars(accelerators, name='z')

            # If there are no incoming requests, terminate ILP
            if sum(s.values()) == 0:
                # if self.profiling_mode is True:
                #     s = {}
                #     for isi in range(num_isi):
                #         s[isi] = 10
                # else:
                self.log.error('No requests received, terminating ILP.')
                return None
            else:
                self.log.debug('\nIncoming requests:' + str(sum(s.values())))
            
            # Set the objective
            m.setObjective((gp.quicksum(w) / sum(s.values()) / 100), GRB.MAXIMIZE)

            if 'accscale' in self.simulator.model_assignment:
                m.addConstr((gp.quicksum(w) / sum(s.values())) >= MINIMUM_ACCURACY, 'c_min_acc')
            # Add constraints
            # m.addConstrs((w[k] == sum(sum(s[k]*b[j, k]*A[j]*z[j, k]*x[i, j] for j in models)
            #                           for i in accelerators) for k in rtypes), 'c1')
            m.addConstr(gp.quicksum(y) / sum(s.values()) >= self.beta, 'c_min_thput')
            for k in rtypes:
                # m.addConstr(w[k] <= sum(sum(s[k]*z[j, k]*b[j, k]*A[j]*x[i, j] for j in models)
                #                         for i in accelerators), 'c1_1_' + str(k))
                # m.addConstr(w[k] == sum(sum(y[j]*b[j, k]*A[j]*x[i, j] for j in models)
                #                         for i in accelerators), 'c1_2_' + str(k))
                # m.addConstr(w[k] == sum(s[k]*z[j,k]*b[j,k]*A[j] for j in models), 'c1_3' + str(k))
                # m.addConstr(w[k] <= sum(sum(y[j]*A[j]*x[i, j] for j in models)
                                        # for i in accelerators), 'c1_4_' + str(k))
                                        
                m.addConstr(w[k] <= sum(sum(s[k]*z[i,k]*x[i,j]*A[j] for j in models) for i in accelerators), 'c1_5_' + str(k))
                # m.addConstr(sum(b[j, k] * z[j, k] for j in models) <= 1, 'c3_' + str(k))
                # m.addConstr(sum(z[j, k] for j in models) <= 1, 'c3_2_' + str(k))

                # m.addConstr(sum(b[j, k] * z[j, k] for j in models) == sum(b[j, k] for j in models), 'c3_' + str(k))
                m.addConstr(sum(sum(b[j, k] * z[i, k] * x[i, j] for j in models) for i in accelerators) == sum(z[i, k] for i in accelerators), 'c3_3k_' + str(k))
                
                m.addConstr(sum(z[i, k] for i in accelerators) == 1, 'c3_2_' + str(k))
                # m.addConstr(sum(z[j, k] * ind[j] for j in models) == 1, 'c_ind_3_' + str(k))

                # raise IlpException('just debugging')
                # m.addConstr(sum(sum(x[i, j] for i in accelerators)* b[j, k] for j in models) >= 1, 'c_no_zeros_' + str(k))

            # if sum(x[i, j] for all i in accelerators) >= 1, then sum(z[j,k] for k in rtypes) > 0

            # m.addConstr(aux * gp.quicksum(y) == 1, 'c_aux')
            # m.addConstr(gp.quicksum(y) >= 1, 'c_bound_1')

            # m.addConstr(aux * gp.quicksum(s) == 1, 'c_aux')

            # m.addConstrs((sum(x[i, j] for j in models) <=
            #              1 for i in accelerators), 'c2')
            for i in accelerators:
                m.addConstr(sum(x[i, j] for j in models) == 1, 'c2_' + str(i))

                m.addConstr(zp[i] == sum(z[i, k] for k in rtypes), 'c_zp_' + str(i))
                m.addConstr(sum(sum(b[j, k] * z[i, k] * x[i, j] for j in models) for k in rtypes) == zp[i], 'c3_3i_' + str(i))

                m.addConstr(y[i] <= sum(p[i, j]*x[i, j] for j in models), 'c4_' + str(i))
                if self.beta <= 1:
                    m.addConstr(y[i] <= sum(z[i, k]*s[k] for k in rtypes), 'c5_' + str(i))
                    # m.addConstr(y[i] <= sum(sum(z[i, k]*s[k]*b[j,k]*x[i,j] for j in models) for k in rtypes),
                    #             'c5_' + str(i))
                else:
                    m.addConstr(y[i] <= self.beta * sum(z[i, k]*s[k] for k in rtypes), 'c5_' + str(i))

                m.addConstr(yp[i] == sum(p[i, j]*x[i, j] for j in models), 'c6_' + str(i))
                m.addConstr(ypp[i] == sum(z[i, k]*s[k] for k in rtypes), 'c7_' + str(i))

                # m.addConstr(10000000 * ind[i] >= sum(x[i, j] for j in models), 'c_ind_1_' + str(i))
                # m.addConstr(ind[i] <= sum(x[i, j] for j in models), 'c_ind_2_' + str(i))

            # * If infeasible, try setting this to =l= 1
            # ct3(k) .. sum(j, b(j,k) * z(j,k)) =e= 1;
            # m.addConstrs((sum(b[j, k] * z[j, k]
            #              for j in models) <= 1 for k in rtypes), 'c3')

                # m.addConstr(y[j] == sum(y_prime[j,k] for k in rtypes))
                # y_prime[j,k] = 

            # As of now, canary_dict plays no role in spec_acc
            # Since model variant can be changed, there is no point of fixing the canary dict at start
            # Though we could constrain canary_dict to split load equally if it does not do so
            # TODO: Check solution to see if load is being split equally as per Sommelier's description
            if self.static is 'spec_acc':
                required_predictors, canary_dict, _ = self.get_solution_from_file(self.starting_allocation)
                m = self.add_spec_acc_constraints(m, x, accelerators, required_predictors)

            # We only do this for AccScale/Proteus
            # if self.simulator.model_assignment == 'ilp' or self.simulator.batching_algo == 'accscale':
            # if self.simulator.model_assignment == 'ilp':
            # m = self.disallow_infeasible_variants(m, x, accelerators,
            #                                       sum(self.simulator.model_variants.values(), []),
            #                                       self.simulator.largest_batch_sizes)

            # # ct4(j) .. y(j) =l= sum(i, x(i,j) * p(i,j));
            # m.addConstrs((y[j] <= sum(p[i, j]*x[i, j]
            #              for i in accelerators) for j in models), 'c4')

            # # ct5(j) .. y(j) =l= sum(k, z(j,k) * s(k));
            # m.addConstrs((y[j] <= sum(z[j, k]*s[k] for k in rtypes)
            #              for j in models), 'c5')

            # Solve ILP
            start_time = time.time()
            m.optimize()
            end_time = time.time()
            ilp_overhead = end_time - start_time
            self.log.info(f'Time to solve ILP: {ilp_overhead} seconds')

            # Measuring routing table overhead for reporting in paper
            # measure_routing_table_overhead()

            if m.status == GRB.OPTIMAL:
                self.log.info('\nObjective: %g' % m.ObjVal)

                total_request_rate = sum(s.values())
                self.log.debug(
                    'Total incoming requests per second:' + str(total_request_rate))

                # throughput = m.ObjVal
                throughput = gp.quicksum(y).getValue()
                normalized_throughput = gp.quicksum(y).getValue() / sum(s.values())
                self.log.debug('Theoretical throughput (objective):' + str(throughput))
                self.log.debug('Percentage of requests met per second (theoretically):' + \
                                str(throughput/total_request_rate*100))
                self.log.info('Normalized throughput from objective: {}'.format(normalized_throughput))
                self.log.info(f'Overall throughput from objective: {throughput}')
                self.log.info(f'Total incoming demand: {sum(s.values())}')
                self.log.info(f'System serving capacity from ILP: {gp.quicksum(yp).getValue()}')
                self.log.info(f'System serving capacity from ILP (ypp): {gp.quicksum(ypp).getValue()}')

                accuracy = gp.quicksum(w).getValue()
                self.log.debug('Accuracy (objective):' + str(accuracy))
                if throughput > 0:
                    self.log.debug('Effective accuracy (over served requests):' + str(accuracy/throughput))
                self.log.debug('Effective accuracy (over all requests):' + str(accuracy/sum(s[k] for k in rtypes)))
                # print('\nx:')
                # for i in accelerators:
                #     for j in models:
                #         if x[i, j].X > 0.0001:
                #             print(f'{i}, {j}, x: {x[i, j].X}')
                self.cached_solution = {}
                self.cached_solution['x'] = x
                self.cached_solution['y'] = y
                self.cached_solution['z'] = z
                self.cached_solution['s'] = s
                self.cached_solution['p'] = p
                self.cached_solution['accelerators'] = accelerators
                self.cached_solution['models'] = models
                self.cached_solution['rtypes'] = rtypes

                effective_accuracy = gp.quicksum(w).getValue()/sum(s.values())
                self.log.info(f'w (effective accuracy): {effective_accuracy}')
                self.simulator.ilp_stats['estimated_effective_accuracy'] = effective_accuracy
                self.simulator.ilp_stats['estimated_throughput'] = throughput
                self.simulator.ilp_stats['demand'] = sum(s.values())
                self.print_cached_solution()
                # actions = self.generate_actions(current_alloc=current_alloc, ilp_solution=x,
                #                                 canary_solution=z, accelerators=accelerators,
                #                                 models=models)
                actions = None
                solution_found = True
                self.beta = self.initial_beta
            else:
                actions = np.zeros(10)
                # self.log.error('No solution')
                if (self.beta > 1.5):
                    decrement = 0.1
                else:
                    decrement = 0.05
                self.log.warn(f'No ILP solution with beta: {self.beta}, trying with: {(self.beta-decrement)}')
                self.beta -= decrement
                # raise IlpException('No solution')
        if solution_found == False:
            raise IlpException(f'No solution found even with beta: {self.beta}')
        
        return actions

    def generate_actions(self, current_alloc: np.ndarray, 
                         ilp_solution: np.ndarray,
                         canary_solution: np.ndarray,
                         accelerators: list,
                         models: list) -> np.ndarray:

        # Generate a 2D np.ndarray from ilp_solution similar to current_alloc
        new_alloc = np.zeros(current_alloc.shape)

        required_predictors = {}
        ilp_x = {}

        for accelerator in accelerators:
            for model_variant in models:
                if ilp_solution[accelerator, model_variant].X == 1:
                    accelerator_type = accelerator.split('-')[0]
                    # new_alloc[j, self.accelerator_dict[accelerator_type]] += 1
                    
                    if (model_variant, accelerator_type) in required_predictors:
                        required_predictors[(model_variant, accelerator_type)] += 1
                    else:
                        required_predictors[(model_variant, accelerator_type)] = 1
                    
                    if (model_variant, accelerator) in ilp_x:
                        ilp_x[(model_variant, accelerator)] += 1
                    else:
                        ilp_x[(model_variant, accelerator)] = 1
        
        canary_dict = {}
        for isi in range(self.num_isi):
            for accelerator in accelerators:
                canary_pct = canary_solution[accelerator, isi].X
                if canary_pct > 0:
                    canary_dict[(accelerator, isi)] = canary_pct
                
                # if canary_pct > 0.0 and canary_pct <= 1.0:
                #     self.log.error(f'accelerator: {accelerator}, isi: {isi}, canary pct: {canary_pct}')

        self.simulator.apply_ilp_solution(required_predictors, canary_dict, ilp_x)
        self.log.debug(f'required_predictors:\n{required_predictors}\ncanary_dict:\n{canary_dict}')

        return None

    def get_cached_solution(self):
        return self.cached_solution

    def print_cached_solution(self):
        if self.cached_solution is None:
            self.log.info('ilp: No solution has been cached yet')
            return
        
        self.log.debug('ilp: Printing cached solution..')
        self.log.debug('\nx:')
        x = self.cached_solution['x']
        y = self.cached_solution['y']
        z = self.cached_solution['z']
        s = self.cached_solution['s']
        p = self.cached_solution['p']
        accelerators = self.cached_solution['accelerators']
        models = self.cached_solution['models']
        rtypes = self.cached_solution['rtypes']

        for i in accelerators:
            self.log.debug(f'y[{i}]: {y[i]}')
            for j in models:
                if x[i, j].X > 0.0001:
                    self.log.debug(f'{i}, {j}, x: {x[i, j].X}, peak throughput: {p[i, j]}')

        for k in rtypes:
            for i in accelerators:
                if z[i, k].X > 0.0001:
                    self.log.debug(f'{i}, {k}, z: {z[i, k].X}')

        for k in rtypes:
            self.log.debug(f's[{k}]: {s[k]}, times beta ({self.initial_beta}): {s[k]*self.initial_beta}')
        return
    
    def measure_routing_table_overhead(self):
        ''' Since routing table lies on the critical path for serving inference requests,
        this function measures the expected overhead of the routing table with a large
        number of entries (order of thousands).
        '''
        request_assignment_dummy = {}
        for dummy in range(5000):
            request_assignment_dummy[dummy] = dummy
        start_time = time.time()
        request_assignment_dummy[50]
        end_time = time.time()
        lookup_overhead = end_time - start_time
        print(f'Routing table lookup overhead: {lookup_overhead}')
        return lookup_overhead
