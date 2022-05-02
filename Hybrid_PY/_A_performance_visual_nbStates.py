import datetime
import matplotlib.pyplot as plt, _0_generic_utils as general_tool
import numpy as np, matplotlib
from itertools import cycle
'''Load'''
results = general_tool.loadJSONFromOutputs("_nbStates_results")
y_test = np.array(results['y_test']) /1000
nbStates_candidates = [5,10,15]
cvrmse_ggmr = []
cvrmse_hybrid = []
for nbStates in nbStates_candidates:
    cvrmse_ggmr.append(results[f'cvrmse_ggmr_{nbStates}'])
    cvrmse_hybrid.append(results[f'cvrmse_hybrid_{nbStates}'])
'''Plot'''
# Jan 29 0:00AM, 2022
large_font = 20
line_width = 4

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

fig, ax = plt.subplots(1)
newline = '\n'
ax.plot(nbStates_candidates, cvrmse_ggmr, next(linecycler), label = "GGMR", linewidth = line_width)
ax.plot(nbStates_candidates, cvrmse_hybrid , next(linecycler),label = "Hybrid", linewidth = line_width)
ax.set_ylabel('CVRMSE %', fontsize=large_font)
ax.set_xlabel('Number of Gaussians', fontsize=large_font)
ax.legend(prop={'size': large_font})
ax.set_title(f'Effect of the number of Gaussains')
ax.tick_params(axis='both', which='major', labelsize=large_font)
plt.show()
pass