import _0_utils as general_tool, datetime, matplotlib.pyplot as plt, numpy as np,statistics
from itertools import cycle
large_font = 15
line_width = 2
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
newline = '\n'

'''Load'''
measure, hybrid,hybrid1, ggmr,ggmr2, ggmr1, rcm3, rcm2,rcm1 = \
    general_tool.load_predicted_measure("all_models_save_hourly_abs.csv")
measure_5min, hybrid_5min,hybrid1_5min, ggmr_5min,ggmr2_5min, ggmr1_5min, rcm3_5min, rcm2_5min,rcm1_5min \
    = general_tool.load_predicted_measure("all_models_save_5min.csv")
'''
Plot
Test start time, Jan 29 0:00AM, 2022
Test duration, 37 days
Time step = hourly
'''
x_date_start = 1643439600
x_date = []
for time_step in range(measure.shape[0]):
    datetime_time = datetime.datetime.fromtimestamp(x_date_start)
    x_date.append(datetime_time)
    x_date_start = x_date_start + 60*60

x_date_start = 1643439600
x_date_5min = []
for time_step in range(measure_5min.shape[0]):
    datetime_time = datetime.datetime.fromtimestamp(x_date_start)
    x_date_5min.append(datetime_time)
    x_date_start = x_date_start + 5*60

fig, ax = plt.subplots(2)

ax[0].plot(x_date, measure/1000, next(linecycler), label = "Measured", linewidth = line_width)
ax[0].plot(x_date, rcm3/1000, next(linecycler),label = "RC Prediction", linewidth = line_width)
ax[0].plot(x_date, ggmr/1000, next(linecycler),label = "GGMR Prediction", linewidth = line_width)
ax[0].plot(x_date, hybrid/1000, next(linecycler),label = "Hybrid Prediction", linewidth = line_width)
ax[0].set_ylabel('Load Power (kW)', fontsize=large_font)
ax[0].legend(prop={'size': large_font})
ax[0].set_title("Prediction Performance (hourly)")

ax[1].plot(x_date_5min, measure_5min/1000, next(linecycler), label = "Measured", linewidth = line_width)
# ax[1].plot(x_date_5min, rcm3_5min/1000, next(linecycler),label = "RC Prediction", linewidth = line_width)
# ax[1].plot(x_date_5min, ggmr_5min/1000, next(linecycler),label = "GGMR Prediction", linewidth = line_width)
# ax[1].plot(x_date_5min, hybrid_5min/1000, next(linecycler),label = "Hybrid Prediction", linewidth = line_width)
ax[1].set_ylabel('Heating Load Power (kW)', fontsize=large_font)
ax[1].legend(prop={'size': large_font})
ax[1].set_title("Prediction Performance (5 mins per step)")
# ax[0].set_title(
#     f'Hybrid approach performance:{newline}'
#     f'NRMSE:{hybrid_nrmse:.2f}%;'
#     f'CVRMSE:{hybrid_cvrmse:.2f}%;'
#     f'MAE:{hybrid_mae/1000:.2f}(kW);'
#     f'MAPE:{hybrid_mape:.2f}%;'
#     f'MdAPE:{hybrid_mdape:.2f}%;'
#     f'gMAPE:{hybrid_gmape:.2f}%.'
#             , fontsize=large_font)
plt.tick_params(axis='both', which='major', labelsize=large_font)

plt.show()
