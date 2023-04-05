import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


SCALE_UP_MULTIPLIER = 1


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inpfile', '-i', required=False, default='../common/twitter_04_14_norm_bursty.txt',
                    dest='inpfile', help='Input file to read. Uses the prep-processed output ' +
                                         'from the INFaaS script')
    ap.add_argument('--distribution', '-d', required=False, default='distribution.txt',
                    dest='dist_file', help='Distribution to follow for dividing requests ' +
                                           'across models')
    ap.add_argument('--slo', '-s', required=False, default=300, dest='slo',
                    help='The SLO value to use for all requests (in milliseconds)')
    ap.add_argument('--out', '-o', required=False, default='../../twitter/asplos/zipf_flat_bursty',
                    dest='out_path', help='Output path to write the traces to')
    return ap.parse_args()


def main(args):
    input_file = args.inpfile
    dist_file = args.dist_file
    out_path = args.out_path
    slo = int(args.slo)

    distribution = []
    num_models = 0
    with open(dist_file, mode='r') as df:
        lines = df.readlines()
        distribution = list(map(lambda x: float(x.rstrip('\n')), lines))
        num_models = len(distribution)
        
    traces = {}
    for model in range(num_models):
        traces[model] = []

    trace = []
    with open(input_file, mode='r') as rf:
        requests_per_second = 400
        seconds = 299

        total_requests = requests_per_second * seconds

        for model in range(num_models):
            requests_for_model = round(total_requests * distribution[model])
            requests_for_model_per_second = requests_for_model / seconds
            
            counter = 0
            arrivals = []
            trace = []
            num_arrivals = 0

            while counter < seconds * 1000:
                arrival = np.rint(np.random.gamma(shape=0.1,
                                                  scale=10000/requests_for_model_per_second))
                trace.append(arrival + counter)
                traces[model].append(arrival + counter)
                counter += arrival
                num_arrivals += 1
                arrivals.append(arrival)
            
            print(f'requests_for_model_per_second: {requests_for_model_per_second}')
            print(f'counter: {counter}, num_arrivals: {num_arrivals}, '
                  f'expected number of requests for this model: {requests_for_model}')
            print(f'requests per second in trace: {num_arrivals / seconds}')
            print()
            continue

    conversions = {}
    conversion_file = '../common/conversion.txt'
    with open(conversion_file, mode='r') as rf:
        lines = rf.readlines()
        conversions = dict(map(lambda x: (x.split(',')[0], x.split(',')[1].rstrip('\n')), lines))

    out_path = os.path.join(out_path, str(slo))
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    for model in range(num_models):
        model_number = str(model + 1)
        model_name = conversions[model_number]
        filename = os.path.join(out_path, f'{model_name}.txt')
        print(filename)

        with open(filename, mode='w') as wf:
            for arrival_time in sorted(traces[model]):
                wf.write(f'{int(arrival_time)},1,{slo}\n')

    print('Done!')


if __name__ == '__main__':
    main(get_args())
