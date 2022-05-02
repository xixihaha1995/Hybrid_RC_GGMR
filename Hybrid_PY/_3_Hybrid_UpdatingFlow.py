import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import _2_all_models as all_predict
import matplotlib.pyplot as plt
import numpy as np, datetime
from itertools import cycle

nbStates = 15
L_rate = 5e-3
training_length = 4032
rc_warming_step = 15

All_Variables_obj, u_measured_obj, abcd = general_tools.switch_case(1)
All_Variables = All_Variables_obj.astype('float64')
u_measured = u_measured_obj.astype('float64')
total_length = All_Variables.shape[1]
test_initial_time = training_length
testing_length = total_length - training_length
# testing_length = 1000
nbVarAll = All_Variables.shape[0]
nbVarInput = nbVarAll - 1

train, test, train_norm, test_norm = general_tools.split_train_test_norm(
    nbVarAll, All_Variables,training_length, testing_length)
center_rc_y, scale_rc_y = train[-2,:].mean(), train[-2,:].std()
'''De-normalization'''
y_test = test[-1,:]
mean_measured = abs(y_test).mean()
center_y, scale_y = train[-1,:].mean(), train[-1,:].std()
'''Hybrid No Flow Information'''
hybrid_flow_norm = all_predict.Hybrid_prediction(train_norm, test_norm, nbStates, nbVarInput,test_initial_time,
                      center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd,L_rate)
hybrid_flow_predict = general_tools.de_norm(hybrid_flow_norm, scale_y, center_y)
cvrmse_hybrid_flow = general_tools.cvrmse_cal(y_test,hybrid_flow_predict,mean_measured)
'''Save results'''
results = {}
results['hybrid_flow_norm'] = np.array(hybrid_flow_norm).reshape(-1).tolist()
results['hybrid_flow_predict'] = hybrid_flow_predict.tolist()
results['cvrmse_hybrid_flow'] = cvrmse_hybrid_flow
results['measured_flow'] = y_test.tolist()
general_tools.saveJSON(results, "_flow_norm_results")
'''plot'''
x_date_start = 1642204800+4032*5*60
x_date = []
for time_step in range(y_test.shape[0]):
    datetime_time = datetime.datetime.fromtimestamp(x_date_start)
    x_date.append(datetime_time)
    x_date_start = x_date_start + 5*60

large_font = 20

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

fig, ax = plt.subplots(1)
newline = '\n'
ax.plot(x_date, y_test, next(linecycler), label = "Measured", )
ax.plot(x_date, hybrid_flow_predict, next(linecycler), label = "Hybrid Flow")
ax.set_ylabel('Volume Flow Rate (CFM)', fontsize=large_font)
ax.legend(prop={'size': large_font})
ax.set_title(f'Hybrid CVRMSE:{cvrmse_hybrid_flow:.2f}%', fontsize=large_font)
ax.tick_params(axis='both', which='major', labelsize=large_font)
plt.show()

