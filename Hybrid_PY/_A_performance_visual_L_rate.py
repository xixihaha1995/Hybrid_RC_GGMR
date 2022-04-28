import datetime
import matplotlib.pyplot as plt, _0_generic_utils as general_tool
import numpy as np, matplotlib
from itertools import cycle
'''Load'''
# _nbStates_results
# _results_t_sigma
# _results_l_rate
results = general_tool.loadJSONFromOutputs("_results_l_rate")
y_test = np.array(results['y_test']) /1000
nbStates_candidates = [5,10,15]
T_Sigma_candidates = [2e-1, 2, 5]
L_rate_candidates = [1e-3,5e-3, 5e-2]
cvrmse_ggmr = []
cvrmse_hybrid = []
for L_rate in L_rate_candidates:
    cvrmse_hybrid.append(results[f'cvrmse_hybrid_{L_rate:.3f}'])
'''Plot'''
# Jan 29 0:00AM, 2022
large_font = 20
line_width = 4

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

fig, ax = plt.subplots(1)
newline = '\n'
ax.plot(L_rate_candidates, cvrmse_hybrid , next(linecycler),label = "Hybrid", linewidth = line_width)
ax.set_ylabel('CVRMSE %', fontsize=large_font)
ax.set_xlabel('Learning rate', fontsize=large_font)
ax.legend(prop={'size': large_font})
ax.set_title(f'Effect of the learning rate')
ax.tick_params(axis='both', which='major', labelsize=large_font)
plt.show()
pass