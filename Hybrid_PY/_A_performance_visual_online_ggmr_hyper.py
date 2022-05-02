import datetime
import matplotlib.pyplot as plt, _0_generic_utils as general_tool
import numpy as np
from itertools import cycle

L_rate_candidates = [8e-2, 1e-1,1.2e-1]
'''Load'''
results = general_tool.loadJSONFromOutputs("_results_online_ggmr")
y_test = np.array(results['y_test']) /1000
rc_y = np.array(results['rc_y']) /1000
cvrmse_rc = results['cvrmse_rc']
cvrmse_ggmr = []
for l_rate in L_rate_candidates:
    cvrmse_ggmr.append(results[f'cvrmse_ggmr_{l_rate:.6f}'])

'''Plot'''
# Jan 29 0:00AM, 2022
x_date_start = 1643439600
x_date = []
for time_step in range(y_test.shape[0]):
    datetime_time = datetime.datetime.fromtimestamp(x_date_start)
    x_date.append(datetime_time)
    x_date_start = x_date_start + 5*60

large_font = 20
line_width = 4

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

fig, ax = plt.subplots(2)
newline = '\n'
ax[0].plot(L_rate_candidates, cvrmse_ggmr, next(linecycler), label = "GGMR", linewidth = line_width)
ax[0].set_ylabel('CVRMSE %', fontsize=large_font)
ax[0].set_xlabel('Learning rate', fontsize=large_font)
ax[0].legend(prop={'size': large_font})
ax[0].set_title(f'Effect of the learning rate')
ax[0].tick_params(axis='both', which='major', labelsize=large_font)

best_ind = np.argmin(cvrmse_ggmr)
best_lrt = L_rate_candidates[best_ind]
ggmr_predict = np.array(results[f'ggmr_predict_{best_lrt:.6f}']) / 1000
best_cvrmse_ggmr = cvrmse_ggmr[best_ind]

ax[1].plot(x_date, y_test, next(linecycler), label = "Measured", linewidth = line_width)
ax[1].plot(x_date, rc_y , next(linecycler),label = "RC", linewidth = line_width)
ax[1].plot(x_date, ggmr_predict,next(linecycler), label = "GGMR", linewidth = line_width)
ax[1].set_ylabel('Load Power (kW)', fontsize=large_font)
ax[1].legend(prop={'size': large_font})
ax[1].set_title(f'RC CVRMSE:{cvrmse_rc:.2f}%{newline}GGMR CVRMSE:{best_cvrmse_ggmr:.2f}%{newline}')
ax[1].tick_params(axis='both', which='major', labelsize=large_font)

plt.show()
pass