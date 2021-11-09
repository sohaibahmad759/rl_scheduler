import os
import random
import argparse
import numpy as np


# TODO: comma-separated list of QoS values [3,4,5,7,7,7,2]. select from this list
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', '-o', required=False,
                    default=os.path.join(os.getcwd(), 'traces/synthetic/newly_generated'),
                    dest='output_dir', help='Directory to write the trace files to. ' +
                    'Default is current directory')
    ap.add_argument('--num_files', '-n', required=True,
                    dest='num_files', help='Number of trace files to generate')
    ap.add_argument('--trace_length', '-l', required=False, default='5000',
                    dest='trace_length', help='Number of requests for every file')
    ap.add_argument('--qos_levels', '-q', required=False, default='1',
                    dest='n_qos_levels', help='Number of QoS levels. Requests are ' +
                    'uniformly distributed over these levels')
    ap.add_argument('--alpha', '-a', required=False, default='10',
                    dest='alpha', help='Alpha value to scale synthetic traces. Default is 10')
    return ap.parse_args()


def main(args):
    trace_path = os.path.join(os.getcwd(), 'traces', args.output_dir)
    trace_length = int(args.trace_length)
    n_qos_levels = int(args.n_qos_levels)
    models = list(range(int(args.num_files)))

    for idx in range(len(models)):
        model = models[idx]

        # we set scale to be proportional to the rank of the model to follow
        # Zipfian distribution
        # scale = 1 / (idx+1)
        scale = int(args.alpha) * (idx+1)

        trace = np.random.exponential(scale=scale, size=trace_length)
        filename = str(model) + '.txt'
        print(filename)

        with open(os.path.join(trace_path, filename), mode='w') as trace_file:
            time = 0
            for arrival in np.rint(trace):
                time += int(arrival)
                trace_file.write(str(time) + ',' + str(random.randint(0, n_qos_levels-1)) + '\n')


if __name__ == '__main__':
    main(get_args())
