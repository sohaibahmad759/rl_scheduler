import glob
import numpy as np
import matplotlib.pyplot as plt


readfiles = glob.glob('cluster_logs/*.txt')
# readfiles = [
            # 'cluster_logs/3b3d80bc931941d7bc3a3458d08960e2.txt',
            # 'cluster_logs/6c35410bb2f14d9d9b17afb92719949b.txt',
            # 'cluster_logs/1537d4d062f3456e99d380cc2225b3d8.txt',
            # 'cluster_logs/dfa82702af7f4c56b57cc42a3ea90223.txt'
            # ]
# readfiles = [
            # '../../logs/per_predictor/300ms/clipper_ht/3b3d80bc931941d7bc3a3458d08960e2.txt',
            # '../../logs/per_predictor/300ms/clipper_ht/6c35410bb2f14d9d9b17afb92719949b.txt',
            # '../../logs/per_predictor/300ms/clipper_ht/1537d4d062f3456e99d380cc2225b3d8.txt',
            # '../../logs/per_predictor/300ms/clipper_ht/dfa82702af7f4c56b57cc42a3ea90223.txt'
            #  ]

requests_per_second = np.zeros(300)
late_per_second = np.zeros(300)

enqueued_requests = 0

for readfile_name in readfiles:
    readfile = open(readfile_name, mode='r')
    lines = readfile.readlines()

    second_counter = 0

    line_counter = 0
    file_finished = False
    times = []
    while line_counter < len(lines):
        line = lines[line_counter]

        if 'process_batch' in line:
            separated = line.strip('\n').split(',')
            if len(separated) > 3:
                simulated = True
                _, time, batchsize, late = separated
            else:
                simulated = False
                _, time, batchsize = separated

            start_time = int(float(time))
            times.append(start_time)
            batchsize = int(batchsize)

            while 'finish_batch_callback' not in line:
                line_counter += 1
                if line_counter >= len(lines):
                    file_finished = True
                    break
                line = lines[line_counter]
                if 'enqueued' in line:
                    enqueued_requests += 1

            if file_finished:
                break

            separated = line.strip('\n').split(',')
            # print(separated)
            if len(separated) > 2:
                _, time, late = separated
            else:
                _, time = separated
            end_time = int(float(time))
            times.append(end_time)

            if late == 'False':
                late = False
            else:
                late = True

            second_counter = int(end_time / 1000)

            print(f'batch size of {batchsize} processed, late: {late}, end_time: '
                f'{end_time}, second_counter: {second_counter}')
            
            requests_per_second[second_counter] += batchsize
            if late:
                late_per_second[second_counter] += batchsize

            if 'finish_batch_callback' not in line:
                raise Exception('next line after process_batch is not finish_batch_callback')
        elif 'enqueued' in line:
            enqueued_requests += 1
        
        line_counter += 1
    if times != sorted(times):
        for i in range(len(times)):
            for j in range(i, len(times)):
                if times[j] < times[i]:
                    print(f'inversion: ({times[i]}, {times[j]}), file: {readfile_name}')
        # print(times)
        raise Exception('file is not sorted by time')

print(requests_per_second)
print(late_per_second)

plt.plot(requests_per_second, label='requests')
plt.plot(late_per_second, label='late')
# plt.savefig(f'{readfile_name}.pdf')
plt.savefig('plot.pdf')
print(f'total enqueued requests: {enqueued_requests}')
print(f'total requests served: {sum(requests_per_second)}')
print(f'total late requests: {sum(late_per_second)}')
print(f'late ratio: {sum(late_per_second)/sum(requests_per_second)}')

