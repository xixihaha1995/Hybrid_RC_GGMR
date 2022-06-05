import _0_utils as general_tool, datetime, matplotlib.pyplot as plt, numpy as np,statistics
from itertools import cycle
large_font = 6
line_width = 4
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
newline = '\n'

'''Load'''
predict, measure = general_tool.load_predicted_measure("hybrid_save.csv")
'''
Plot
Test start time, Jan 29 0:00AM, 2022
Test duration, 37 days
Time step = hourly
'''
predict, measure = predict[:37*24], measure[:37*24]
errors_dist = general_tool.all_error(measure,predict)
hybrid_nrmse = general_tool.nrmse(measure,predict)
hybrid_cvrmse = general_tool.cv_rmse(measure, predict)
hybrid_mae = general_tool.mae(measure, predict)
hybrid_mape = general_tool.mean_absolute_percentage_error(measure, predict)
hybrid_mdape = general_tool.median_absolute_percentage_error(measure, predict)
hybrid_gmape = general_tool.geometric_median_absolute_percentage_error(measure, predict)

x_date_start = 1643439600
x_date = []
for time_step in range(measure.shape[0]):
    datetime_time = datetime.datetime.fromtimestamp(x_date_start)
    x_date.append(datetime_time)
    x_date_start = x_date_start + 60*60

fig, ax = plt.subplots(3)

ax[2].plot(x_date, errors_dist)
ax[2].set_ylabel("Percentage errors (x100%)")

three_quartiles = statistics.quantiles(errors_dist, n=4)
print(f'The three quartiles of the percentage errors:{three_quartiles}')

bin_width = 0.01
bin_x = np.arange(bin_width, 1 + bin_width, bin_width)
ax[1].hist(errors_dist, bins=bin_x,
           weights=np.ones(len(errors_dist)) / len(errors_dist))
ax[1].set_xlabel(f"Percentage errors (unit = -, stepsize={bin_width})")
ax[1].set_ylabel("Percentage error occurance probability")
ax[0].plot(x_date, measure/1000, next(linecycler), label = "Measured", linewidth = line_width)
ax[0].plot(x_date, predict/1000, next(linecycler),label = "Hybrid Predicted", linewidth = line_width)
ax[0].set_ylabel('Heating Load Power (kW)', fontsize=large_font)
ax[0].legend(prop={'size': large_font})
ax[0].set_title(
    f'Hybrid approach performance:{newline}'
    f'NRMSE:{hybrid_nrmse:.2f}%;'
    f'CVRMSE:{hybrid_cvrmse:.2f}%;'
    f'MAE:{hybrid_mae/1000:.2f}(kW);'
    f'MAPE:{hybrid_mape:.2f}%;'
    f'MdAPE:{hybrid_mdape:.2f}%;'
    f'gMAPE:{hybrid_gmape:.2f}%.'
            , fontsize=large_font)
plt.tick_params(axis='both', which='major', labelsize=large_font)

plt.show()
