import _1_gmr_ggmr_hybrid_utils as gaussian_tools
import numpy as np


'''GGMR'''
def GGMR_prediction(train_norm, test_norm, nbStates, T_sigma, L_rate = 5e-3):
    hybrid = True
    if not hybrid:
        train_norm_ggmr = np.delete(train_norm, -2, axis=0) #delete rc_y information
        test_norm_ggmr = np.delete(test_norm, -2, axis=0) #delete rc_y information
    else:
        train_norm_ggmr = train_norm
        test_norm_ggmr = test_norm
    nbVarAll_ggmr = train_norm_ggmr.shape[0]
    nbVarInput_ggmr = nbVarAll_ggmr - 1

    init_Priors_ggmr, init_Mu_ggmr, init_Sigma_ggmr = gaussian_tools.EM_Init_Func(train_norm_ggmr, nbStates)
    em_Priors_ggmr, em_Mu_ggmr, em_Sigma_ggmr  = gaussian_tools.EM_Func(
        train_norm_ggmr, init_Priors_ggmr, init_Mu_ggmr, init_Sigma_ggmr)
    unused_ggmr_method_gmr_data, ggmr_beta = gaussian_tools.GMR_Func(
        em_Priors_ggmr, em_Mu_ggmr, em_Sigma_ggmr, test_norm_ggmr[:nbVarInput_ggmr,:], nbVarInput_ggmr)
    sum_beta_rs_ggmr=sum(ggmr_beta,1).reshape(1,-1)


    ggmr_norm = gaussian_tools.ggmr_func(em_Priors_ggmr, em_Mu_ggmr, em_Sigma_ggmr,
                                         test_norm_ggmr,sum_beta_rs_ggmr,L_rate,T_sigma)
    return ggmr_norm

'''Hybrid'''
def Hybrid_prediction(train_norm, test_norm, nbStates, nbVarInput,test_initial_time,
                      center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd,L_rate,T_Sigma):
    init_Priors_gmr, init_gmr, init_Sigma_gmr = gaussian_tools.EM_Init_Func(train_norm, nbStates)
    em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr  = gaussian_tools.EM_Func(train_norm, init_Priors_gmr, init_gmr, init_Sigma_gmr )
    gmr_norm, gmr_beta = gaussian_tools.GMR_Func(em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr, test_norm[:nbVarInput,:], nbVarInput)
    sum_beta_rs=sum(gmr_beta,1).reshape(1,-1)

    hybrid_norm = gaussian_tools.hybrid_func(em_Priors_gmr, em_Mu_gmr, em_Sigma_gmr,
                                             test_norm,sum_beta_rs,test_initial_time, center_rc_y,
                                             scale_rc_y,u_measured, rc_warming_step,abcd,L_rate,T_Sigma)
    return hybrid_norm


