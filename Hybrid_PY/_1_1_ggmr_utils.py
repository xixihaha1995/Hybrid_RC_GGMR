import numpy as np
import _1_gmr_ggmr_hybrid_utils as gauss_tools_one
from sklearn import preprocessing

def online_ggmr_new_dominate(train_norm, test_norm,train, test,
                             max_nbStates, lrn_rate,
                look_back_batch_size, predict_size,_hybrid, T_merge, plonum):

    if not _hybrid:
        train_norm = np.delete(train_norm, -2, axis=0) # delete rc_y information
        test_norm = np.delete(test_norm, -2, axis=0)  # delete rc_y information
        train = np.delete(train, -2, axis=0) # delete rc_y information
        test = np.delete(test, -2, axis=0)  # delete rc_y information

    old_prior, old_mu, old_sigma = gauss_tools_one.online_init(train_norm, max_nbStates)

    nbVar = test_norm.shape[0]
    in_out_split = nbVar - 1
    _look_back_batch_size = look_back_batch_size
    _predict_size = predict_size
    expData = np.array([])
    for t_stamp in range(0, test_norm.shape[1], _predict_size):
        print(t_stamp)
        if t_stamp > _look_back_batch_size:
            _batch_prev_norm = test_norm[:, t_stamp - _look_back_batch_size: t_stamp]
            new_init_nb_com = min(_look_back_batch_size,2)

            Priors_init, Mu_init, Sigma_init = gauss_tools_one.EM_Init_Func(_batch_prev_norm, new_init_nb_com, True)
            new_prior, new_mu, new_sigma = gauss_tools_one.EM_Func(_batch_prev_norm, Priors_init, Mu_init, Sigma_init)

            old_prior, old_mu, old_sigma = gauss_tools_one.merge_new_into_old(t_stamp, _look_back_batch_size,
                                                              _batch_prev_norm,
                                                              max_nbStates,
                                                              old_prior, old_mu, old_sigma,
                                                              new_prior, new_mu, new_sigma,
                                                              policy_num = plonum ,lrn_rate =lrn_rate, t_merge = T_merge)

        _batch = test_norm[:, t_stamp: t_stamp + _predict_size]

        this_exp_y_norm, dummy_Gaus_weights = gauss_tools_one.GMR_Func(old_prior, old_mu, old_sigma ,
                                                  _batch[:in_out_split,:], in_out_split)
        expData = np.concatenate((expData, this_exp_y_norm.reshape(-1)))
    return expData
