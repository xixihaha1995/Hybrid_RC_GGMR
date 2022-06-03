import datetime
import matplotlib.pyplot as plt, _0_generic_utils as general_tool
import numpy as np, matplotlib
from itertools import cycle
'''Load'''
results = general_tool.loadJSONFromOutputs("_results_initial_ggmr_boucha_best")
y_test = np.array(results['y_test']) /1000
rc_y = np.array(results['rc_y']) /1000
ggmr_predict = np.array(results['ggmr_predict']) /1000
# hybrid_predict = np.array(results['hybrid_predict']) /1000
cvrmse_rc = results['cvrmse_rc']
cvrmse_ggmr =results['cvrmse_ggmr']
# cvrmse_hybrid = results['cvrmse_hybrid']

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

fig, ax = plt.subplots(1)
newline = '\n'
ax.plot(x_date, y_test, next(linecycler), label = "Measured", linewidth = line_width)
ax.plot(x_date, rc_y , next(linecycler),label = "RC", linewidth = line_width)
ax.plot(x_date, ggmr_predict,next(linecycler), label = "GGMR", linewidth = line_width)
# ax.plot(x_date, hybrid_predict, next(linecycler), label = "Hybrid", linewidth = line_width)
ax.set_ylabel('Load Power (kW)', fontsize=large_font)
ax.legend(prop={'size': large_font})
ax.set_title(f'RC CVRMSE:{cvrmse_rc:.2f}%{newline}GGMR CVRMSE:{cvrmse_ggmr:.2f}%{newline}'
             , fontsize=large_font)
# font = {'family' : 'normal',
#         'size'   : 22}
# matplotlib.rc('font', **font)
ax.tick_params(axis='both', which='major', labelsize=large_font)
plt.show()
pass