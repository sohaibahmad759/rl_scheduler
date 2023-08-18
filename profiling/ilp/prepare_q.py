import os
import shutil


trace_path = '../../traces/twitter/asplos/zipf_exponential/profiling/'
starter_trace = f'{trace_path}/starter_trace/bert.txt'

maximum_q = 30

for i in range(1, maximum_q+1):
    for j in range(1, i+1):
        output_path = os.path.join(trace_path, 'q', str(i), f'bert_{j}.txt')

        shutil.copy2(starter_trace, output_path)
    print(f'Copied trace files to {output_path}/q/{i}')
