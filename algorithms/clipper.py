import json
from algorithms.base import SchedulingAlgorithm


class Clipper(SchedulingAlgorithm):
    def __init__(self, simulator):
        SchedulingAlgorithm.__init__(self, 'Clipper')

        self.simulator = simulator
        self.solution_file = 'algorithms/clipper_solution_highacc.txt'
        self.solution_applied = False

        return

    def apply_solution(self):
        ''' Applies the solution from self.solution_file
        '''
        if self.solution_applied:
            return

        with open(self.solution_file, mode='r') as rf:
            lines = rf.readlines()
            required_predictors = eval(lines[1].rstrip('\n'))
            canary_dict = eval(lines[3].rstrip('\n'))

            print(f'required_predictors: {required_predictors}')
            print(f'canary_dict: {canary_dict}')
            self.simulator.apply_ilp_solution(required_predictors, canary_dict)
            self.solution_applied = True

        return
