import numpy as np
import _1_gmr_ggmr_hybrid_utils as gauss_tools_one

def online_ggmr_new_dominate(train_norm, test_norm,max_nbStates, lrn_rate,
                look_back_batch_size, predict_size,_hybrid, std_bool, plonum):
    if not _hybrid:
        train_norm = np.delete(train_norm, -2, axis=0) # delete rc_y information
        test_norm = np.delete(test_norm, -2, axis=0)  # delete rc_y information

    # scaler = preprocessing.StandardScaler().fit(train.T)
    # train_scaled = scaler.transform(train.T)
    # old_center_y, old_scale_y = scaler.mean_[-1], scaler.scale_[-1]

    old_prior, old_mu, old_sigma = gauss_tools_one.online_init(train_norm, max_nbStates)

    nbVar = test_norm.shape[0]
    in_out_split = nbVar - 1
    _look_back_batch_size = look_back_batch_size
    _predict_size = predict_size
    expData = np.array([])
    for t_stamp in range(0, test_norm.shape[1], _predict_size):
        print(t_stamp)
        if t_stamp > _look_back_batch_size:
            _batch_prev = test_norm[:, t_stamp - _look_back_batch_size: t_stamp]
            _batch_next_norm = test_norm[:, t_stamp: t_stamp + _predict_size]
            # scaler = preprocessing.StandardScaler().fit(_batch_prev.T)
            # _batch_prev_scaled = scaler.transform(_batch_prev.T)
            # old_center_y, old_scale_y = scaler.mean_[-1], scaler.scale_[-1]

            Priors_init, Mu_init, Sigma_init = gauss_tools_one.EM_Init_Func(_batch_prev, _look_back_batch_size, True)
            new_prior, new_mu, new_sigma = gauss_tools_one.EM_Func(_batch_prev, Priors_init, Mu_init, Sigma_init)

            old_prior, old_mu, old_sigma = gauss_tools_one.merge_new_into_old(t_stamp, _look_back_batch_size,
                                                              _batch_prev, _batch_next_norm,
                                                              max_nbStates,
                                                              old_prior, old_mu, old_sigma,
                                                              new_prior, new_mu, new_sigma,
                                                              policy_num = plonum ,lrn_rate =lrn_rate, t_merge = None)

        _batch = test_norm[:, t_stamp: t_stamp + _predict_size]
        this_exp_y_norm, dummy_Gaus_weights = gauss_tools_one.GMR_Func(old_prior, old_mu, old_sigma ,
                                                  _batch[:in_out_split,:], in_out_split)
        expData = np.concatenate((expData, this_exp_y_norm.reshape(-1)))
    return expData