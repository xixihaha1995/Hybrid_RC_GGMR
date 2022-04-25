'''
TODO: Load inputs
GMR
GGMR
Hybrid
'''
import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import matplotlib.pyplot as plt, os, scipy.io as sio
import numpy as np
nbStates = 15


All_Variables_obj, u_measured_obj, abcd_obj = general_tools.switch_case(0)
All_Variables = All_Variables_obj.astype('float64')
u_measured = u_measured_obj.astype('float64')
abcd = abcd_obj

total_length = All_Variables.shape[1]
training_length = 4032
test_initial_time = training_length -1
rc_warming_step = 14
testing_length = total_length - training_length
# testing_length = 1000
talk_to_rc  = 0

nbVarAll = All_Variables.shape[0]
nbVarInput = nbVarAll - 1

train, test, train_norm, test_norm = general_tools.split_train_test_norm(
    nbVarAll, All_Variables,training_length, testing_length)

'''GMR'''

# init_Priors_gmr, init_gmr, init_Sigma_gmr = gaussian_tools.EM_Init_Func(train_norm, nbStates)
# em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr  = gaussian_tools.EM_Func(train_norm, init_Priors_gmr, init_gmr, init_Sigma_gmr )
# gmr_norm, gmr_beta = gaussian_tools.GMR_Func(em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr, test_norm[:nbVarInput,:], nbVarInput)

'''GGMR'''
center_rc_y, scale_rc_y = train[-2,:].mean(), train[-2,:].std()
train_norm_ggmr = np.delete(train_norm, -2, axis=0) #delete rc_y information
test_norm_ggmr = np.delete(test_norm, -2, axis=0) #delete rc_y information
nbVarAll_ggmr = train_norm_ggmr.shape[0]
nbVarInput_ggmr = nbVarAll_ggmr - 1

init_Priors_ggmr, init_Mu_ggmr, init_Sigma_ggmr = gaussian_tools.EM_Init_Func(train_norm_ggmr, nbStates)
em_Priors_ggmr, em_Mu_ggmr, em_Sigma_ggmr  = gaussian_tools.EM_Func(
    train_norm_ggmr, init_Priors_ggmr, init_Mu_ggmr, init_Sigma_ggmr)
unused_ggmr_method_gmr_data, ggmr_beta = gaussian_tools.GMR_Func(
    em_Priors_ggmr, em_Mu_ggmr, em_Sigma_ggmr, test_norm_ggmr[:nbVarInput_ggmr,:], nbVarInput_ggmr)

sum_beta_rs_ggmr=sum(ggmr_beta,1).reshape(1,-1)
ggmr_talk_rc = 0
L_rate = 5e-3

script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','evolving_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

em_Priors_ggmr = mat_contents['rs_Priors_ggmr']
em_Mu_ggmr = mat_contents['rs_Mu_ggmr']
em_Sigma_ggmr = mat_contents['rs_Sigma_ggmr']
test_norm_ggmr = mat_contents['test_norm_ggmr']
sum_beta_rs_ggmr = mat_contents['sum_beta_rs_ggmr']
ggmr_talk_rc = mat_contents['ggmr_talk_rc']
test_initial_time = mat_contents['test_initial_time']
center_rc_y = mat_contents['center_rc_y']
scale_rc_y = mat_contents['scale_rc_y']
u_measured = mat_contents['u_measured']
rc_warming_step = mat_contents['rc_warming_step']
abcd = mat_contents['abcd']
L_rate = mat_contents['L_rate']

ggmr_norm = gaussian_tools.Evolving_LW_2_Func(em_Priors_ggmr, em_Mu_ggmr, em_Sigma_ggmr,
                                                             test_norm_ggmr,sum_beta_rs_ggmr, ggmr_talk_rc,
                                                             test_initial_time, center_rc_y, scale_rc_y,
                                                             u_measured, rc_warming_step,abcd,L_rate)

'''De-normalization'''
y_test = test[-1,:]
rc_y = test[-2,:]
mean_measured = abs(y_test).mean()
center_y, scale_y = train[-1,:].mean(), train[-1,:].std()

# gmr_predict = gmr_norm * scale_y + center_y
# gmr_predict = gmr_predict.reshape(-1)
# rmse_gmr = (sum((y_test - gmr_predict) ** 2) / len(y_test)) ** (1 / 2)
# cvrmse_gmr = rmse_gmr * 100 / mean_measured

rmse_rc= (sum((y_test - rc_y) ** 2) / len(y_test)) ** (1 / 2)
cvrmse_rc = rmse_rc*100 / mean_measured

ggmr_norm_tmp = np.array(ggmr_norm).reshape(-1)
ggmr_predict = ggmr_norm_tmp * scale_y + center_y
ggmr_predict = ggmr_predict.reshape(-1)
rmse_ggmr = (sum((y_test - ggmr_predict) ** 2) / len(y_test)) ** (1 / 2)
cvrmse_ggmr = rmse_ggmr * 100 / mean_measured


fig, ax = plt.subplots(1)
newline = '\n'
ax.plot(y_test, label = "Measured")
ax.plot(rc_y, label = "RC")
ax.plot(ggmr_predict, label = "GGMR")
ax.legend()
ax.set_title(f'RCCVRMSE:{cvrmse_rc}%{newline}GGMR CVRMSE:{cvrmse_ggmr}% ')
plt.show()

pass

