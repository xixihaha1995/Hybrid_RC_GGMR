'''
TODO: Load inputs
GMR
GGMR
Hybrid
'''
import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import matplotlib.pyplot as plt, os, scipy.io as sio
import numpy as np


'''GGMR'''
def GGMR_prediction(train_norm, test_norm, nbStates):
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

    ggmr_norm = gaussian_tools.Evolving_LW_2_Func(em_Priors_ggmr, em_Mu_ggmr, em_Sigma_ggmr,
                                                                 test_norm_ggmr,sum_beta_rs_ggmr, ggmr_talk_rc,
                                                                 test_initial_time, center_rc_y, scale_rc_y,
                                                                 u_measured, rc_warming_step,abcd,L_rate)
    return ggmr_norm

'''Hybrid'''
init_Priors_gmr, init_gmr, init_Sigma_gmr = gaussian_tools.EM_Init_Func(train_norm, nbStates)
em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr  = gaussian_tools.EM_Func(train_norm, init_Priors_gmr, init_gmr, init_Sigma_gmr )
gmr_norm, gmr_beta = gaussian_tools.GMR_Func(em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr, test_norm[:nbVarInput,:], nbVarInput)

sum_beta_rs=sum(gmr_beta,1).reshape(1,-1)

hybrid_norm = gaussian_tools.Evolving_LW_2_Func(em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr,
                                                test_norm,sum_beta_rs,real_time_talk_to_rc,
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

hybrid_norm_tmp = np.array(hybrid_norm).reshape(-1)
hybrid_predict = hybrid_norm_tmp * scale_y + center_y
hybrid_predict = hybrid_predict.reshape(-1)
rmse_hybrid = (sum((y_test - hybrid_predict) ** 2) / len(y_test)) ** (1 / 2)
cvrmse_hybrid = rmse_hybrid * 100 / mean_measured

fig, ax = plt.subplots(1)
newline = '\n'
ax.plot(y_test, label = "Measured")
ax.plot(rc_y, label = "RC")
ax.plot(ggmr_predict, label = "GGMR")
ax.plot(hybrid_predict, label = "Hybrid")
ax.legend()
ax.set_title(f'RC CVRMSE:{cvrmse_rc}%{newline}GGMR CVRMSE:{cvrmse_ggmr}%{newline}'
             f'Hybrid CVRMSE:{cvrmse_hybrid}%')
plt.show()

pass

