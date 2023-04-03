import matplotlib.pyplot as plt


# trace = 'twitter_04_14_norm'
trace = 'twitter_04_14_norm_bursty'

requests = []
with open(f'{trace}.txt', mode='r') as rf:
    for line in rf:
        timestep_requests = int(line.split()[1].rstrip('\n'))
        requests.append(timestep_requests)

plt.plot(requests)
plt.savefig(f'{trace}.pdf')
