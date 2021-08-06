import os
from heapq import merge

trace_path = 'traces/synthetic/'

trace_files = os.listdir(trace_path)
out_file_name = 'traces/synthetic/combined.txt'

trace_dict = {}

for readfile in trace_files:
    isi_name, extension = readfile.split('.')
    if not extension == 'txt':
        continue
    requests = []
    print(readfile)
    with open(os.path.join(trace_path, readfile), mode='r') as rf:
        for line in rf:
            requests.append(int(line.rstrip('\n')))
    trace_dict[isi_name] = requests

final_arr = []

for key in trace_dict:
    print(key)
    arr = trace_dict[key]
    final_arr = list(merge(final_arr, arr))

with open(out_file_name, mode='w') as out_file:
    for item in final_arr:
        out_file.write(str(item) + '\n')
