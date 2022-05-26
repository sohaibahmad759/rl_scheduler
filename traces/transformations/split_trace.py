trace_name = 'efficientnet'
rfilename = trace_name + '.txt'

split_factor = 2
wfilenames = []
wfiles = []

for i in range(split_factor):
    wfilename = trace_name + '_' + str(i+1) + '.txt'
    wfilenames.append(wfilename)
    wf = open(wfilename, 'w')
    wfiles.append(wf)

counter = 0
with open(rfilename, mode='r') as rf:
    for line in rf:
        if counter == split_factor:
            counter = 0
        
        wfiles[counter].write(line)

        counter += 1