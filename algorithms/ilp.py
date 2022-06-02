import time
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from algorithms.base import SchedulingAlgorithm


class Ilp(SchedulingAlgorithm):
    def __init__(self, allocation_window):
        SchedulingAlgorithm.__init__(self, 'ILP')

        self.log = logging.getLogger(__name__)

        self.log.addHandler(logging.FileHandler('logs/ilp/output.log', mode='w'))
        self.log.setLevel(logging.DEBUG)

        self.allocation_window = allocation_window

        self.simulator = None

        # logging.basicConfig(filename='output.log',
        #                     mode='w',
        #                     format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
        #                     level=logging.DEBUG)

    def is_simulator_set(self):
        if self.simulator is None:
            return True
        else:
            return False

    def set_simulator(self, simulator):
        self.simulator = simulator

    def run(self, observation, num_acc_types, num_max_acc):
        # print('observation shape:', observation.shape)
        # print('observation:', observation)

        num_isi = observation.shape[0] - 1
        # For simple use cases, the number of models = number of ISIs
        num_models = num_isi
        
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
            # print(isi)
            rtypes.append(isi)

            # Initialize demand for each request type (ISI)
            s[isi] = demand[isi]

        # We want to randomly initialize accuracy values for now, but we set the
        # seed so that accuracy values are always the same
        np.random.seed(0)

        for model in range(num_models):
            models.append(model)

            A[model] = np.random.randint(50, 100)

            for accelerator in accelerators:
                accelerator_type = accelerator.split('-')[0]
                latency = latencies[isi, self.accelerator_dict[accelerator_type]]

                if latency is None:
                    throughput = 0
                else:
                    throughput = 1000 / latency
                p[accelerator, model] = throughput

            for isi in range(num_isi):
                if model == isi:
                    b[model, isi] = 1
                else:
                    b[model, isi] = 0

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
        y = m.addVars(models, name='y')
        z = m.addVars(jk_pairs, name='z')
        aux = m.addVar(name='aux')

        # Smaller value weighs throughput more, higher value weighs accuracy more
        # alpha = 0.0
        alpha = 0.9

        # If there are no incoming requests, terminate ILP
        if sum(s.values()) == 0:
            self.log.error('No requests received, terminating ILP.')
            return None
        else:
            print('\nIncoming requests:' + str(sum(s.values())))

        # TODO: Accuracy is from [0, 100] but throughput can be [0, +inf]
        # TODO: We can bound throughput by dividing it by total number of requests to make throughput %
        # TODO: That way, they will also both be on the same scale (i.e., [0, 100])
        # Set the objective
        # m.setObjective(alpha * gp.quicksum(w) * aux + (1-alpha) * gp.quicksum(y) / sum(s.values()), GRB.MAXIMIZE)
        m.setObjective(alpha * gp.quicksum(w) / sum(s.values()) / 100 + (1-alpha) * gp.quicksum(y) / sum(s.values()), GRB.MAXIMIZE)

        # Add constraints
        # m.addConstrs((w[k] == sum(sum(s[k]*b[j, k]*A[j]*z[j, k]*x[i, j] for j in models)
        #                           for i in accelerators) for k in rtypes), 'c1')
        for k in rtypes:
            # m.addConstr(w[k] <= sum(sum(s[k]*z[j, k]*b[j, k]*A[j]*x[i, j] for j in models)
            #                         for i in accelerators), 'c1_1_' + str(k))
            # m.addConstr(w[k] == sum(sum(y[j]*b[j, k]*A[j]*x[i, j] for j in models)
            #                         for i in accelerators), 'c1_2_' + str(k))
            
            # TODO: weird thing happens where if w[k] becomes important, all accelerators go to same model
            # m.addConstr(w[k] == sum(s[k]*z[j,k]*b[j,k]*A[j] for j in models), 'c1_3' + str(k))
            # m.addConstr(w[k] <= sum(sum(y[j]*A[j]*x[i, j] for j in models)
                                    # for i in accelerators), 'c1_4_' + str(k))
            m.addConstr(w[k] == sum(y[j]*z[j,k]*A[k] for j in models), 'c1_5_' + str(k))
            # m.addConstr(sum(b[j, k] * z[j, k] for j in models) <= 1, 'c3_' + str(k))
            # m.addConstr(sum(z[j, k] for j in models) <= 1, 'c3_2_' + str(k))
            m.addConstr(sum(b[j, k] * z[j, k] for j in models) == sum(b[j, k] for j in models), 'c3_' + str(k))
            m.addConstr(sum(z[j, k] for j in models) == 1, 'c3_2_' + str(k))

        # m.addConstr(aux * gp.quicksum(y) == 1, 'c_aux')
        # m.addConstr(gp.quicksum(y) >= 1, 'c_bound_1')

        m.addConstr(aux * gp.quicksum(s) == 1, 'c_aux')

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

        # # ct4(j) .. y(j) =l= sum(i, x(i,j) * p(i,j));
        # m.addConstrs((y[j] <= sum(p[i, j]*x[i, j]
        #              for i in accelerators) for j in models), 'c4')

        # # ct5(j) .. y(j) =l= sum(k, z(j,k) * s(k));
        # m.addConstrs((y[j] <= sum(z[j, k]*s[k] for k in rtypes)
        #              for j in models), 'c5')

        # Solve ILP
        m.optimize()

        if m.status == GRB.OPTIMAL:
            self.log.info('\nObjective: %g' % m.ObjVal)

            total_request_rate = sum(s.values())
            self.log.debug(
                'Total incoming requests per second:' + str(total_request_rate))

            # throughput = m.ObjVal
            throughput = gp.quicksum(y).getValue()
            self.log.debug('Theoretical throughput (objective):' + str(throughput))
            self.log.debug('Percentage of requests met per second (theoretically):' + \
                            str(throughput/total_request_rate*100))

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
                                            accelerators=accelerators, models=models)
        else:
            actions = np.zeros(current_alloc.shape)
            self.log.error('No solution')
        
        return actions

    def generate_actions(self, current_alloc: np.ndarray, 
                         ilp_solution: np.ndarray,
                         accelerators: list,
                         models: list) -> np.ndarray:

        # Generate a 2D np.ndarray from ilp_solution similar to current_alloc
        new_alloc = np.zeros(current_alloc.shape)

        for i in accelerators:
            for j in models:
                if ilp_solution[i, j].X == 1:
                    accelerator_type = i.split('-')[0]
                    new_alloc[j, self.accelerator_dict[accelerator_type]] += 1

        # print('new allocation:', new_alloc)
        self.log.debug('current_allocation:' + str(current_alloc))

        # Take a difference with current_alloc

        # Return suggested actions from the difference matrix
        return new_alloc


if __name__ == "__main__":
    # Test out if the inheritance works fine
    x = Ilp()
    x.print_algorithm()
    y = SchedulingAlgorithm()
    y.print_algorithm()
