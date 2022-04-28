import datetime
import matplotlib.pyplot as plt, _0_generic_utils as general_tool
import numpy as np, matplotlib
from itertools import cycle
'''Load'''


results = general_tool.loadJSONFromOutputs("_results_t_sigma")
y_test = np.array(results['y_test']) /1000
nbStates_candidates = [5,10,15]
T_Sigma_candidates = [2e-1, 2, 5]
cvrmse_ggmr = []
cvrmse_hybrid = []
for t_sigma in T_Sigma_candidates:
    cvrmse_ggmr.append(results[f'cvrmse_ggmr_{t_sigma:.2f}'])
    cvrmse_hybrid.append(results[f'cvrmse_hybrid_{t_sigma:.2f}'])
'''Plot'''
# Jan 29 0:00AM, 2022
large_font = 20
line_width = 4

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

fig, ax = plt.subplots(1)
newline = '\n'
ax.plot(T_Sigma_candidates, cvrmse_ggmr, next(linecycler), label = "GGMR", linewidth = line_width)
ax.plot(T_Sigma_candidates, cvrmse_hybrid , next(linecycler),label = "Hybrid", linewidth = line_width)
ax.set_ylabel('CVRMSE %', fontsize=large_font)
ax.set_xlabel('Closeness threshold', fontsize=large_font)
ax.legend(prop={'size': large_font})
ax.set_title(f'Effect of the closeness threshold')
ax.tick_params(axis='both', which='major', labelsize=large_font)
plt.show()
pass