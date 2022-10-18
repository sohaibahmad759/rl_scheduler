import os
from numpy import arange
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = 'accscale_vs_infaas.pdf'

thput_our = [99.914, 98.77, 98.69, 94.14, 92.4, 87.3]
acc_our = [72.61937473, 79.53659519,
           79.55678269, 79.86759679, 79.91766141, 80.07]

thput_infaas = [90.67,
                87.48,
                68.92]
acc_infaas = [73.6908206,
           76.78,
           77.39849862]


# def objective(x, a, b, c):
# 	return a * x + b * x**2 + c

# popt, pcov = curve_fit(objective, acc_our, thput_our)

# print(popt)

# x_line = arange(min(acc_our), max(acc_our), 1)
# # calculate the output for the range
# y_line = objective(x_line, popt[0], popt[1], popt[2])

# plt.plot(x_line, y_line)

plt.plot(acc_our, thput_our, label='AccScale', marker='^')
# plt.scatter(acc_our, thput_our)
plt.plot(acc_infaas, thput_infaas, label='INFaaS', marker='v')
# plt.scatter(acc_infaas, thput_infaas)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2)
plt.xlabel('Effective Accuracy (%)')
plt.ylabel('Normalized Throughput (%)')
plt.grid(linestyle='--')
plt.xticks(arange(70, 83, 2))
plt.yticks(arange(60, 101, 5))
# plt.show()
plt.savefig(os.path.join('..', 'figures', 'paper_draft_figures', filename), dpi=500)

