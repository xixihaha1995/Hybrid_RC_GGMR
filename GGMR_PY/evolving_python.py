import _utils
import numpy as np


def Evolving_LW_2(Priors, Mu, Sigma, Data_Test,SumPosterior,talk_to_rc, test_initial_time,
    center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd, L_rate):
    C_mat = SumPosterior.T;
    nbVar = Data_Test.shape[0]
    in_out_split = nbVar - 1
    expData = []
    for t in range(1, Data_Test.shape[1]):
        this_exp_y, dummy_Gaus_weights = _utils.GMR_Func(Priors, Mu, Sigma, Data_Test[:in_out_split, t], in_out_split)
        expData.append(this_exp_y)
        [Priors, Mu, Sigma, C_mat] = ggmr_update_gaussian(Data_Test,Priors, Mu, Sigma, t, C_mat, L_rate)

    return expData

def ggmr_update_gaussian(Data_Test,Priors, Mu, Sigma, t, C_mat, L_rate):
    T_sigma = 2
    eps_thres_best_priors = 1e-2
    pumax = 0.09
    existFlag_up_sig = 0

    m_best, Post_pr = _utils.BMC_Func(Data_Test[:, t], Priors, Mu, Sigma, )
    com_MD_lst = []

    for m in range(Priors.shape[1]):
        this_com_MD = _utils.Mahal_dis_Func(Data_Test[:, t], Mu[:, m], Sigma[:, :, m])
        com_MD_lst.append(this_com_MD)
    com_MD = np.array(com_MD_lst).reshape(-1, 1)

    if (com_MD[m_best, 0] < T_sigma) and (Post_pr[m_best, 0] > eps_thres_best_priors):
        existFlag_up_sig = 1

    if existFlag_up_sig != 1:
        pass
        '''
        Only update one best Gaussian
        '''
        q_j = Post_pr[m_best, 0] / np.sum(Post_pr, axis= 0 )
        C_mat[m_best, 0] += q_j
        tau_j = (1 - L_rate) @ Priors[m_best, 0] + L_rate @ q_j
        Priors[m_best, 0] = min(tau_j, pumax)
        eta_j = q_j @ ((1 - L_rate) / C_mat[m_best, 0] + L_rate)
        Mu[:, m_best] = (1 - eta_j) * Mu[:, m_best] + eta_j * Data_Test[:,t]
        Sigma[:,:,m_best] = (1 - eta_j) * Sigma[:,:,m_best] + eta_j * \
                            ((Data_Test[:,t] - Mu[:, m_best]) @ (Data_Test[:,t] - Mu[:, m_best]).T)

        Priors = Priors / np.sum(Post_pr, axis= 0 )

    return [Priors, Mu, Sigma, C_mat]








