import _0_utils as general_tool, datetime, matplotlib.pyplot as plt, numpy as np,statistics
from itertools import cycle
large_font = 6
line_width = 4
lines = ["-","--","-.",":"]
linecycler = cycle(lines)
newline = '\n'

'''Load'''
measure, hybrid, hybrid1, ggmr,ggmr2, ggmr1, rcm3, rcm2,rcm1,\
abs_measure, abs_hybrid, abs_hybrid1, abs_ggmr, abs_ggmr2, abs_ggmr1, abs_rcm3, abs_rcm2, abs_rcm1\
    =general_tool.load_all("all_models_save_hourly_ori_abs.csv")
'''
Plot
Test start time, Jan 29 0:00AM, 2022
Test duration, 37 days
Time step = hourly
'''
measure = measure[:37*24]
abs_measure = abs_measure[:37*24]
all_predictions = {}
all_predictions['hybrid'] = hybrid
all_predictions['hybrid1'] = hybrid1
all_predictions['ggmr'] = ggmr
all_predictions['ggmr2'] = ggmr2
all_predictions['ggmr1'] = ggmr1
all_predictions['rc_model3'] = rcm3
all_predictions['rc_model2'] = rcm2
all_predictions['rc_model1'] = rcm1

for model_name, predict in all_predictions.items():
    predict = predict[:37*24]
    errors_dist = general_tool.all_error(measure,predict)
    nrmse = general_tool.nrmse_final(measure,predict,abs_measure)
    cvrmse = general_tool.cv_rmse_final(measure, predict, abs_measure)
    mae = general_tool.mae(measure, predict)
    mape = general_tool.mean_absolute_percentage_error_final(measure, predict,abs_measure)

    print(f'Performance for model:{model_name}{newline}'
          f'nrmse:{nrmse:.2f}%,cvrmse:{cvrmse:.2f}%,'
          f'mae:{mae/1000:.2f}(kW),mape:{mape:.2f}%.')

