import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


SCALE_UP_MULTIPLIER = 1


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inpfile', '-i', required=False, default='../common/twitter_04_14_norm.txt',
                    dest='inpfile', help='Input file to read. Uses the prep-processed output ' +
                                         'from the INFaaS script')
    ap.add_argument('--distribution', '-d', required=False, default='distribution.txt',
                    dest='dist_file', help='Distribution to follow for dividing requests ' +
                                           'across models')
    ap.add_argument('--slo', '-s', required=False, default=300, dest='slo',
                    help='The SLO value to use for all requests (in milliseconds)')
    ap.add_argument('--out', '-o', required=False, default='../../twitter/asplos/zipf_uniform',
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
        overall_offset = 0
        for line in rf:
            # the number of requests for a given second
            requests = int(int(line.split()[1].rstrip('\n')) * SCALE_UP_MULTIPLIER)
            second = line.split()[0]
            print(f'second: {second}, requests: {requests}')

            for model in range(num_models):
                offset = overall_offset
                requests_for_model = round(requests * distribution[model])
                
                if requests_for_model == 0:
                    continue

                # we need to distribute these requests around the entire second using the exponential
                # distribution for inter-arrival rates. In other words, the requests follow Poisson
                # arrival within the second
                # arrivals = np.rint(np.random.uniform(high=requests_for_model,
                #                                      size=requests_for_model))
                arrivals = np.ones(requests_for_model)
                arrivals = (arrivals / sum(arrivals) * 1000).astype(int)
                print(sum(arrivals))
                print(arrivals)
                if sum(arrivals) > 1000:
                    print(requests_for_model)
                    print('oops')
                    exit()
                # plt.hist(arrivals)
                # plt.savefig('uniform_arrivals.pdf')
                # plt.close()
                # exit()

                current = 0
                # print(arrivals)
                for time in arrivals:
                    current += time
                    if time < 1000:
                        traces[model].append(offset + current)

                # at the end of the second, we increment the offset in milliseconds
                offset += 1000
            overall_offset = offset

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
