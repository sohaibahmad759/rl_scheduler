import time
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from algorithms.base import SchedulingAlgorithm


class Ilp(SchedulingAlgorithm):
    def __init__(self, allocation_window, alpha):
        SchedulingAlgorithm.__init__(self, 'ILP')

        self.log = logging.getLogger(__name__)

        self.log.addHandler(logging.FileHandler('logs/ilp/output.log', mode='w'))
        self.log.setLevel(logging.DEBUG)

        self.allocation_window = allocation_window

        self.alpha = alpha

        self.simulator = None
        self.num_isi = 0

        # logging.basicConfig(filename='output.log',
        #                     mode='w',
        #                     format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
        #                     level=logging.DEBUG)

    def is_simulator_set(self):
        if self.simulator is None:
            return False
        else:
            return True

    def set_simulator(self, simulator):
        self.simulator = simulator

    def run(self, observation, num_acc_types, num_max_acc):
        # print('observation shape:', observation.shape)
        # print('observation:', observation)

        num_isi = observation.shape[0] - 1
        # For simple use cases, the number of models = number of ISIs
        num_models = num_isi
        self.num_isi = num_isi
        
        current_alloc = observation[0:num_isi, 0:num_acc_types]

        demand_since_last = observation[0:num_isi, -2]
        # divide demand by time elapsed since last measurement to get demand in units of requests per second
        demand = demand_since_last / (self.allocation_window / 1000)
        missed_requests = observation[0:num_isi, -1]

        latencies = observation[0:num_isi, num_acc_types:2*num_acc_types]

        # models = range(num_isi)

        # First generate the parameter variables from the isi_dict
        rtypes = []

        models = []

        # Openvino only runs on CPUs with Intel AVX capability. All nodes are AVX capable
        # CPUs for OnnxRuntime and OpenVino overlap 100%
        # Just have a CPU type
        accelerators = []
        for acc in range(num_max_acc):
            accelerators.append('CPU-' + str(acc))
            accelerators.append('GPU_AMPERE-' + str(acc))
            accelerators.append('VPU-' + str(acc))
            accelerators.append('GPU_PASCAL-' + str(acc))

        self.accelerator_dict = {'CPU': 0, 'GPU_AMPERE': 1, 'VPU': 2, 'GPU_PASCAL': 3}

        # accelerators = [NodeModelAcceleratorType.ONNXRUNTIME_GPU_PASCAL,
        #                 NodeModelAcceleratorType.ONNXRUNTIME_GPU_AMPERE,
        #                 NodeModelAcceleratorType.ONNXRUNTIME_CPU,
        #                 NodeModelAcceleratorType.OPENVINO_CPU]

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
        # TODO: Initialize this
        b = {}

        for isi in range(num_isi):
            rtypes.append(isi)

            # Initialize demand for each request type (ISI)
            s[isi] = demand[isi]

            isi_name = self.simulator.idx_to_executor[isi]
            model_variants = self.simulator.model_variants[isi_name]

            for model_variant in model_variants:
                models.append(model_variant)

                # A[model_variant] = np.random.randint(50, 100)
                A[model_variant] = self.simulator.model_variant_accuracies[(
                    isi_name, model_variant)]

                for accelerator in accelerators:
                    accelerator_type = accelerator.split('-')[0]
                    latency = latencies[isi, self.accelerator_dict[accelerator_type]]

                    if latency is None:
                        throughput = 0
                    else:
                        throughput = 1000 / latency
                    p[accelerator, model_variant] = throughput

                for isi in range(num_isi):
                    # if model == isi:
                    #     b[model, isi] = 1
                    # else:
                    #     b[model, isi] = 0
                    if model_variant in self.simulator.model_variants[self.simulator.idx_to_executor[isi]]:
                        b[model_variant, isi] = 1
                    else:
                        b[model_variant, isi] = 0

        # Helper variables for setting up the ILP
        ij_pairs, _ = gp.multidict(p)
        jk_pairs, _ = gp.multidict(b)

        # Set up the optimization model
        m = gp.Model('Accuracy Throughput MILP')

        m.setParam('NonConvex', 2)
        m.setParam('TimeLimit', 200)
        m.setParam('MIPGap', 0.5)

        # Add optimization variables
        w = m.addVars(rtypes, name='x')
        x = m.addVars(ij_pairs, vtype=GRB.BINARY, name='x')
        # TODO: Both these variables need to be positive
        # y = m.addVars(models, name='y', vtype=GRB.POSITIVE)
        # z = m.addVars(jk_pairs, name='z', vtype=GRB.POSITIVE)
        y = m.addVars(models, name='y')
        ind = m.addVars(models, name='ind', vtype=GRB.BINARY)
        z = m.addVars(jk_pairs, name='z')
        # aux = m.addVar(name='aux')

        # Smaller value weighs throughput more, higher value weighs accuracy more
        # alpha = 0.0
        alpha = self.alpha

        # If there are no incoming requests, terminate ILP
        if sum(s.values()) == 0:
            self.log.error('No requests received, terminating ILP.')
            return None
        else:
            print('\nIncoming requests:' + str(sum(s.values())))
        
        # TODO: how to set accelerators for each model variant (and specify model variant while doing so)?
        # Set the objective
        # m.setObjective(alpha * gp.quicksum(w) * aux + (1-alpha) * gp.quicksum(y) / sum(s.values()), GRB.MAXIMIZE)
        m.setObjective((alpha * gp.quicksum(w) / sum(s.values()) / 100) + ((1-alpha) * gp.quicksum(y) / sum(s.values())), GRB.MAXIMIZE)

        # Add constraints
        # m.addConstrs((w[k] == sum(sum(s[k]*b[j, k]*A[j]*z[j, k]*x[i, j] for j in models)
        #                           for i in accelerators) for k in rtypes), 'c1')
        m.addConstr(gp.quicksum(y) / sum(s.values()) >= 0.8, 'c_min_thput')
        for k in rtypes:
            # m.addConstr(w[k] <= sum(sum(s[k]*z[j, k]*b[j, k]*A[j]*x[i, j] for j in models)
            #                         for i in accelerators), 'c1_1_' + str(k))
            # m.addConstr(w[k] == sum(sum(y[j]*b[j, k]*A[j]*x[i, j] for j in models)
            #                         for i in accelerators), 'c1_2_' + str(k))
            # m.addConstr(w[k] == sum(s[k]*z[j,k]*b[j,k]*A[j] for j in models), 'c1_3' + str(k))
            # m.addConstr(w[k] <= sum(sum(y[j]*A[j]*x[i, j] for j in models)
                                    # for i in accelerators), 'c1_4_' + str(k))
            # TODO: how of many of the requests in y[j] belong to isi k?
            # m.addConstr(w[k] <= sum(y[j]*z[j,k]*A[j] for j in models), 'c1_5_' + str(k))
            # TODO: Probably this constraint is forcing z[j,k] to be either 1 or 0
            #       When alpha = 0, z[j,k] varies from 0 to 1 quite often
            m.addConstr(w[k] <= sum(s[k]*z[j,k]*A[j] for j in models), 'c1_5_' + str(k))
            # m.addConstr(sum(b[j, k] * z[j, k] for j in models) <= 1, 'c3_' + str(k))
            # m.addConstr(sum(z[j, k] for j in models) <= 1, 'c3_2_' + str(k))

            # TODO: Jun 1: model is infeasible due to this particular constraint, without it the model
            #               works but gives weird result
            # TODO: edit: it seems like this issue has been resolved with constraint c3_3_
            # m.addConstr(sum(b[j, k] * z[j, k] for j in models) == sum(b[j, k] for j in models), 'c3_' + str(k))
            m.addConstr(sum(b[j, k] * z[j, k] for j in models) == sum(z[j,k] for j in models), 'c3_3_' + str(k))
            m.addConstr(sum(z[j, k] for j in models) <= 1, 'c3_2_' + str(k))
            # m.addConstr(sum(z[j, k] * ind[j] for j in models) == 1, 'c_ind_3_' + str(k))

            m.addConstr(sum(sum(x[i, j] for i in accelerators)*b[j, k]
                        for j in models) >= 1, 'c_no_zeros_' + str(k))

        # if sum(x[i, j] for all i in accelerators) >= 1, then sum(z[j,k] for k in rtypes) > 0

        # m.addConstr(aux * gp.quicksum(y) == 1, 'c_aux')
        # m.addConstr(gp.quicksum(y) >= 1, 'c_bound_1')

        # m.addConstr(aux * gp.quicksum(s) == 1, 'c_aux')

        # m.addConstrs((sum(x[i, j] for j in models) <=
        #              1 for i in accelerators), 'c2')
        for i in accelerators:
            m.addConstr(sum(x[i, j] for j in models) <= 1, 'c2_' + str(i))

        # * If infeasible, try setting this to =l= 1
        # ct3(k) .. sum(j, b(j,k) * z(j,k)) =e= 1;
        # m.addConstrs((sum(b[j, k] * z[j, k]
        #              for j in models) <= 1 for k in rtypes), 'c3')

        for j in models:
            m.addConstr(y[j] <= sum(p[i, j]*x[i, j] for i in accelerators), 'c4_' + str(j))
            # TODO: z[j,k] summed over j is <= 1.
            # this constraint here is summing z[j,k] over k
            # This could still be correct, just need to make sure
            m.addConstr(y[j] <= sum(z[j, k]*s[k] for k in rtypes), 'c5_' + str(j))

            m.addConstr(100000 * ind[j] >= sum(x[i, j] for i in accelerators), 'c_ind_1_' + str(j))
            m.addConstr(ind[j] <= sum(x[i, j] for i in accelerators), 'c_ind_2_' + str(j))

            # m.addConstr(y[j] == sum(y_prime[j,k] for k in rtypes))
            # y_prime[j,k] = 

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
        print(f'Time to solve ILP: {ilp_overhead} seconds')

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
            self.log.debug('Normalized throughput from objective: {}'.format(normalized_throughput))

            accuracy = gp.quicksum(w).getValue()
            self.log.debug('Accuracy (objective):' + str(accuracy))
            if throughput > 0:
                self.log.debug('Effective accuracy (over served requests):' + str(accuracy/throughput))
            self.log.debug('Effective accuracy (over all requests):' + str(accuracy/sum(s[k] for k in rtypes)))
            # self.log.debug('\nx:')
            # for i in accelerators:
            #     for j in models:
            #         if x[i, j].X > 0.0001:
            #             self.log.debug('{}, {}, x: {}'.format(i, j, x[i, j].X))
            actions = self.generate_actions(current_alloc=current_alloc, ilp_solution=x,
                                            canary_solution=z, accelerators=accelerators,
                                            models=models)
        else:
            actions = np.zeros(current_alloc.shape)
            self.log.error('No solution')
        
        return actions

    def generate_actions(self, current_alloc: np.ndarray, 
                         ilp_solution: np.ndarray,
                         canary_solution: np.ndarray,
                         accelerators: list,
                         models: list) -> np.ndarray:

        # Generate a 2D np.ndarray from ilp_solution similar to current_alloc
        new_alloc = np.zeros(current_alloc.shape)

        required_predictors = {}

        for accelerator in accelerators:
            for model_variant in models:
                if ilp_solution[accelerator, model_variant].X == 1:
                    accelerator_type = accelerator.split('-')[0]
                    # new_alloc[j, self.accelerator_dict[accelerator_type]] += 1
                    
                    # TODO: How to remove predictors currently in the system?
                    # TODO: We are going to add predictors manually
                    #       Maybe similar to BLIS implementation, we can build a dictionary first
                    if (model_variant, accelerator_type) in required_predictors:
                        required_predictors[(model_variant, accelerator_type)] += 1
                    else:
                        required_predictors[(model_variant, accelerator_type)] = 1
        
        canary_dict = {}
        for isi in range(self.num_isi):
            for model_variant in models:
                canary_pct = canary_solution[model_variant, isi].X
                if canary_pct > 0:
                    canary_dict[(model_variant, isi)] = canary_pct
                
                if canary_pct > 0.0 and canary_pct < 1.0:
                    logging.info('variant: {}, isi: {}, canary pct: {}'.format(model_variant, isi, canary_pct))


        # logging.info('required_predictors: {}'.format(required_predictors))
        # logging.info('canary dict: {}'.format(canary_dict))
        # time.sleep(1)
        self.simulator.apply_predictor_dict(required_predictors, canary_dict)

        return None

        # print('new allocation:', new_alloc)
        self.log.debug('current_allocation:' + str(current_alloc))

        # Take a difference with current_alloc

        # Return suggested actions from the difference matrix
        return new_alloc
    
    def measure_routing_table_overhead(self):
        ''' Since routing table lies on the critical path for serving inference requests,
        this function measures the expected overhead of the routing table with a large
        number of entries (order of thousands).
        '''
        # request_assignment_dummy = {'resnet1': 0.3, 'resnet2': 0.2, 'resnet3': 0.3,
        #                             'resnet4': 0.3, 'resnet5': 0.2, 'resnet6': 0.3,
        #                             'resnet7': 0.3, 'resnet8': 0.2, 'resnet9': 0.3,
        #                             'resnet50': 0.3, 'resnet18': 0.2, 'resnet34': 0.3}
        request_assignment_dummy = {}
        for dummy in range(5000):
            request_assignment_dummy[dummy] = dummy
        start_time = time.time()
        request_assignment_dummy[50]
        end_time = time.time()
        lookup_overhead = end_time - start_time
        print('Routing table lookup overhead:', lookup_overhead)


if __name__ == "__main__":
    # Test out if the inheritance works fine
    x = Ilp()
    x.print_algorithm()
    y = SchedulingAlgorithm()
    y.print_algorithm()
