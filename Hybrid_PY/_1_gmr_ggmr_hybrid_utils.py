import sys, numpy as np, copy, math, _0_generic_utils as general_tools
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def gaussPDF_Func(Data_ori, Mu, Sigma):
    if Data_ori.ndim == 1:
        nbVar, nbData = Data_ori.shape[0], 1
    else:
        nbVar, nbData  = Data_ori.shape
    Data = Data_ori.T - np.tile(Mu.T, [nbData, 1])
    prob = np.sum(Data @ np.linalg.inv(Sigma) * Data, axis=1)
    prob = np.exp(-0.5 * prob )/ np.sqrt((2 * np.pi) ** nbVar * (abs(np.linalg.det(Sigma)) + sys.float_info.min) )
    return prob

def EM_Init_Func(Data, nbStates, kmean_init_=False):
    Data_tran = Data.T
    nbVar = Data_tran.shape[1]
    if not kmean_init_:
        minc = np.min(Data_tran, axis=0)
        maxc = np.max(Data_tran, axis=0)
        all_var_ran = []
        for idx_var in range(nbVar):
            step = (maxc[idx_var] - minc[idx_var]) / (nbStates)
            if step == 0:
                this_var_ran = np.array([minc[idx_var] for _ in range(nbStates)])
            else:
                this_var_ran = np.arange(minc[idx_var], maxc[idx_var] - 1e-5, step)
            all_var_ran.append(this_var_ran)
        all_var_cen = np.array(all_var_ran).T
        kmeans = KMeans(n_clusters=nbStates, init=all_var_cen,random_state=0).fit(Data_tran)
    else:
        kmeans = KMeans(n_clusters=nbStates, algorithm="elkan").fit(Data_tran)
    Mu = kmeans.cluster_centers_.T
    Priors_lst = []
    Sigma_lst = []
    for cluster_idx in range(nbStates):
        Priors_lst.append(Data_tran[np.where(kmeans.labels_ == cluster_idx)].shape[0])
        this_cluster_samps = Data_tran[np.where(kmeans.labels_ == cluster_idx)].T
        if this_cluster_samps.shape[1] == 1:
            this_cluster_sigma = np.zeros([this_cluster_samps.shape[0], this_cluster_samps.shape[0]]) \
                                 + 1e-5 * np.identity(nbVar)
        else:
            this_cluster_sigma = np.cov(this_cluster_samps) + 1e-5*np.identity(nbVar)
        Sigma_lst.append(this_cluster_sigma)
    Priors = np.array(Priors_lst) / np.sum(Priors_lst).reshape(1,-1)
    Sigma = np.array(Sigma_lst).T
    return Priors, Mu, Sigma

def EM_Func(Data, Priors0, Mu0, Sigma0):
    loglik_threshold = 1e-10
    (nbVar, nbData) = Data.shape
    nbStates = Sigma0.shape[2]
    loglik_old = - sys.float_info.max
    nbStep = 0
    Mu = copy.deepcopy(Mu0)
    Sigma = copy.deepcopy(Sigma0)
    Priors =copy.deepcopy(Priors0)
    while 1:
        pass
        '''E-step'''
        Pxi_lst = []
        for i in range(nbStates):
            # compute probability p(x|i)
            this_px = gaussPDF_Func(Data, Mu[:,i],Sigma[:,:,i])
            Pxi_lst.append(this_px)
        Pxi = np.array(Pxi_lst).reshape(-1, nbStates)
        # compute posterior probability p(i|x)
        Pix_tmp = np.tile(Priors,[nbData, 1]) * Pxi
        denom = np.tile(np.sum(Pix_tmp, axis=1).reshape(-1, 1), [1, nbStates])
        denom[denom == 0] = sys.float_info.min
        Pix_nan = Pix_tmp /denom

        Pix = np.nan_to_num(Pix_nan, nan=0)
        # compute the cumulated posterior probability
        E = np.sum(Pix, axis = 0).reshape(1,-1) + sys.float_info.min
        '''M-step'''
        for i in range(nbStates):
            Priors[0,i] = E[0,i] / nbData
            Mu[:,i] = Data @ Pix[:,i] / E[0,i]
            Data_tmp1 = Data - np.tile(Mu[:, i].reshape(-1,1), [1, nbData])
            Sigma[:,:, i] = (np.tile(Pix[:, i].T,[nbVar, 1]) * Data_tmp1 @ Data_tmp1.T) / E[0,i] + 1e-5*np.identity(nbVar)
        '''Stopping criterion'''
        Pxi_lst = []
        for i in range(nbStates):
            # compute probability p(x|i)
            this_px = gaussPDF_Func(Data, Mu[:, i], Sigma[:, :, i])
            Pxi_lst.append(this_px)
        Pxi = np.array(Pxi_lst).reshape(-1, nbStates)
        F_nan = Pxi @ Priors.T
        F = np.nan_to_num(F_nan, nan=sys.float_info.min)
        F[F==0] = sys.float_info.min
        loglik = np.log(F).mean()
        # print(abs((loglik/loglik_old)-1))

        if abs((loglik / loglik_old) - 1) < loglik_threshold\
                or nbStep > 100:
            break
        loglik_old = loglik
        nbStep += 1

    Sigma[:, :, :] += 1e-5 * np.identity(nbVar).reshape(nbVar,nbVar,-1)
    return Priors, Mu, Sigma

def GMR_Func(Priors, Mu, Sigma, input_x, in_out_split):
    nbVar = Mu.shape[0]
    nbVarInput = nbVar - 1
    if input_x.ndim == 1:
        temp, nbData = input_x.shape[0], 1
    else:
        [temp, nbData] = input_x.shape

    nbStates = Sigma.shape[2]

    Px = []
    for i in range(nbStates):
        this_Pxi = Priors[0, i] * gaussPDF_Func(input_x, Mu[:in_out_split,i], Sigma[:in_out_split, :in_out_split, i])
        Px.append(this_Pxi)
    Px_reshape = np.array(Px).T
    beta = Px_reshape / np.tile(np.sum(Px_reshape, axis= 1).reshape(-1,1) + sys.float_info.min,[1, nbStates])

    y_temp_lst = []
    for j in range(nbStates):
        this_y_tmp = np.tile(Mu[in_out_split:, j], [1, nbData]) + Sigma[in_out_split:,:in_out_split, j] \
                     @ np.linalg.inv(Sigma[:in_out_split,:in_out_split, j]) @ \
                     (input_x - np.tile(Mu[:in_out_split, j].reshape(-1,1),[1, nbData]))
        y_temp_lst.append(np.array(this_y_tmp).T.reshape(-1))

    y_tmp = np.array(y_temp_lst).T.reshape(-1,nbData,nbStates)
    beta_tmp = beta.reshape(-1,nbData,nbStates)
    y_tmp2 = np.tile(beta_tmp,[nbVar - in_out_split, 1,1]) * y_tmp
    y = np.sum(y_tmp2, axis=2)
    return y, beta

def BMC_Func(Data_in,Priors_in,Mu_in,Sigma_in):
    Post_pr_lst =[]
    for m in range(Priors_in.shape[1]):
        this_post_pr = Priors_in[0, m].reshape(1) @ gaussPDF_Func(Data_in, Mu_in[:,m], Sigma_in[:,:,m]) + sys.float_info.min
        Post_pr_lst.append(this_post_pr)
    Post_pr = np.array(Post_pr_lst).reshape(-1,1)
    m_best = np.argmax(Post_pr)
    return m_best, Post_pr

def Mahal_dis_Func(Data,Mu,Cov):
    pass
    # Md = sqrt(abs((Data - Mu)'*inv(Cov)*(Data-Mu)));
    Md = np.sqrt(abs((Data - Mu).T @ np.linalg.inv(Cov) @ (Data - Mu)))
    return Md

def ggmr_create_update_gaussian(Data_Test,Priors, Mu, Sigma, t, C_mat, L_rate, T_sigma):
    # T_sigma = 2
    # eps_thres_best_priors = 1e-6
    eps_thres_best_priors = 1e-2
    tau_min_thres = 0.09
    existFlag_sig = 0
    k_o_init_sigma = 300

    m_best, Post_pr = BMC_Func(Data_Test[:, t], Priors, Mu, Sigma)
    com_MD_lst = []

    for m in range(Priors.shape[1]):
        this_com_MD = Mahal_dis_Func(Data_Test[:, t], Mu[:, m], Sigma[:, :, m])
        com_MD_lst.append(this_com_MD)
    com_MD = np.array(com_MD_lst).reshape(-1, 1)

    if (com_MD[m_best, 0] < T_sigma) and (Post_pr[m_best, 0] > eps_thres_best_priors):
        existFlag_sig = 1

    if existFlag_sig == 0:
        '''create new gaussian'''
        pass
        print("create new")
        for nb_com in range(Priors.shape[1]):
            Priors[0, nb_com] = (1- L_rate) * Priors[0, nb_com]
        # _least_contr_gau = np.argmin(Priors)

        Priors = np.hstack((Priors, np.array([[L_rate]])))
        C_mat = np.vstack((C_mat, 1))
        Mu = np.hstack((Mu, Data_Test[:, t].reshape(-1,1)))
        Sigma = np.vstack((Sigma.T, k_o_init_sigma*np.identity(Sigma.shape[0])[None]))
        Sigma = Sigma.T


    if existFlag_sig == 1:
        '''
        Update all Gaussians
        '''
        print("Updating")
        for nb_com in range(Post_pr.shape[0]):
            q_j = Post_pr[nb_com, 0] / np.sum(Post_pr, axis= 0 )
            C_mat[nb_com, 0] += q_j
            tau_j =( (1 - L_rate) * Priors[0,nb_com] + L_rate * q_j)[0]
            Priors[0,nb_com] = min(tau_j, tau_min_thres)
            eta_j = q_j * ((1 - L_rate) / C_mat[nb_com, 0] + L_rate)
            eta_j = np.nan_to_num(eta_j, nan= L_rate)
            eta_j = eta_j[0]
            Mu[:, nb_com] = (1 - eta_j) * Mu[:, nb_com] + eta_j * Data_Test[:,t]
            x_min_mu = (Data_Test[:, t] - Mu[:, nb_com]).reshape(-1, 1)
            update_sigma = (1 - eta_j) * Sigma[:,:,nb_com] + eta_j * x_min_mu @ x_min_mu.T
            Sigma[:,:,nb_com] = update_sigma

    return [Priors, Mu, Sigma, C_mat]

def hybrid_update_gaussian(Data_Test,Priors, Mu, Sigma, t, C_mat, L_rate, T_sigma):
    # T_sigma = 2
    # eps_thres_best_priors = 1e-6
    eps_thres_best_priors = 1e-2
    tau_min_thres = 0.09
    existFlag_sig = 0

    m_best, Post_pr = BMC_Func(Data_Test[:, t], Priors, Mu, Sigma)
    com_MD_lst = []

    for m in range(Priors.shape[1]):
        this_com_MD = Mahal_dis_Func(Data_Test[:, t], Mu[:, m], Sigma[:, :, m])
        com_MD_lst.append(this_com_MD)
    com_MD = np.array(com_MD_lst).reshape(-1, 1)

    if (com_MD[m_best, 0] < T_sigma) and (Post_pr[m_best, 0] > eps_thres_best_priors):
        existFlag_sig = 1

    if existFlag_sig == 0:
        '''
        Old Update strategy: No best gaussian -> Update all gaussians
        Update all Gaussians
        '''
        print("Updating")
        for nb_com in range(Post_pr.shape[0]):
            q_j = Post_pr[nb_com, 0] / np.sum(Post_pr, axis= 0 )
            C_mat[nb_com, 0] += q_j
            tau_j =( (1 - L_rate) * Priors[0,nb_com] + L_rate * q_j)[0]
            Priors[0,nb_com] = min(tau_j, tau_min_thres)
            eta_j = q_j * ((1 - L_rate) / C_mat[nb_com, 0] + L_rate)
            eta_j = np.nan_to_num(eta_j, nan= L_rate)
            eta_j = eta_j[0]
            Mu[:, nb_com] = (1 - eta_j) * Mu[:, nb_com] + eta_j * Data_Test[:,t]
            x_min_mu = (Data_Test[:, t] - Mu[:, nb_com]).reshape(-1, 1)
            update_sigma = (1 - eta_j) * Sigma[:,:,nb_com] + eta_j * x_min_mu @ x_min_mu.T
            Sigma[:,:,nb_com] = update_sigma

    return [Priors, Mu, Sigma, C_mat]

def split_func(Priors, Mu, Sigma,C_mat, t_split_fac, time_stam, max_nbStates):
    split_factor = 8e-1
    # can volume be negative
    cannot_merge_link = -1
    largst_comp = -1
    # if max(abs(np.linalg.det(Sigma.T))) < 0:
    #     print("Sigma volumes can be negative")
    max_volumes = max(abs(np.linalg.det(Sigma.T)))
    mean_volumes = abs(np.linalg.det(Sigma.T)).mean()
    # print(f'max_volumes:{max_volumes}, mean_volumes:{mean_volumes}')
    if max_volumes < t_split_fac * mean_volumes or Priors.shape[1] >= max_nbStates:
        return Priors, Mu, Sigma, C_mat, cannot_merge_link, largst_comp
    print("Spliting")
    cannot_merge_link = time_stam
    largst_comp = np.argmax(np.linalg.det(Sigma.T))
    lamdas, eigen_vecs = np.linalg.eig(Sigma.T[largst_comp,:,:])
    dpc_idx = np.argmax(lamdas)
    lamda = lamdas[dpc_idx]
    eigen_vec = eigen_vecs[:, dpc_idx]
    delta_eigen_vect = (split_factor * lamda)**(0.5) * eigen_vec
    tau_prev = Priors[0, largst_comp]
    tau_one = tau_two =  np.array([[tau_prev / 2]])
    mu_prev = Mu[:, largst_comp]
    # c_mat_prev = C_mat[largst_comp,0]
    c_mat_one = c_mat_two = 1
    mu_one = mu_prev + delta_eigen_vect
    mu_two = mu_prev - delta_eigen_vect
    sigma_one = sigma_two = Sigma.T[largst_comp,:,:] - delta_eigen_vect @ delta_eigen_vect.T

    Priors[0, largst_comp] = copy.deepcopy(tau_one)
    Mu[:, largst_comp] = copy.deepcopy(mu_one)
    Sigma[:,:,largst_comp] = copy.deepcopy(sigma_one)
    C_mat[largst_comp,0] = c_mat_one

    Priors = np.hstack((Priors, tau_two))
    Mu = np.hstack((Mu, mu_two.reshape(-1,1)))
    Sigma = np.vstack((Sigma.T,sigma_two[None]))
    Sigma = Sigma.T
    C_mat = np.vstack((C_mat, c_mat_two))
    return Priors, Mu, Sigma, C_mat, cannot_merge_link, largst_comp

def skld_func(sig_one, sig_two, mu_one, mu_two):
    D = sig_one.shape[0]
    kld_one = 1/2 * (np.log(abs(np.linalg.det(sig_one) / np.linalg.det(sig_two))) \
              + np.trace(np.linalg.inv(sig_two) @ sig_one) \
              + (mu_two - mu_one).reshape(-1,1).T @ np.linalg.inv(sig_one) @ (mu_two - mu_one).reshape(-1,1) - D)
    kld_two =1/2 * (np.log(abs(np.linalg.det(sig_two) / np.linalg.det(sig_one))) \
              + np.trace(np.linalg.inv(sig_one) @ sig_two) \
              + (mu_one - mu_two).reshape(-1,1).T @ np.linalg.inv(sig_two) @ (mu_one - mu_two).reshape(-1,1) - D)
    skld = 1/2 * (kld_one + kld_two)
    return skld

def merge_func(Priors, Mu, Sigma,C_mat,t_merge_fac, cannot_merge_link, largst_comp):
    skld_dict = {}
    nbComp = Sigma.shape[-1]
    if nbComp <= 3:
        return Priors, Mu, Sigma, C_mat
    for ind_i in range(nbComp):
        for ind_j in range(ind_i + 1, nbComp):
            sig_A, sig_B = Sigma[:,:,ind_i],Sigma[:,:,ind_j]
            mu_A, mu_B = Mu[:,ind_i], Mu[:,ind_j]
            this_skld = skld_func(sig_A, sig_B, mu_A, mu_B)
            skld_dict[(f'{ind_i}',f'{ind_j}')] = abs(this_skld[0,0])
    ind_one, ind_two = min(skld_dict, key = skld_dict.get)
    ind_one, ind_two = int(ind_one), int(ind_two)

    min_skld = min(skld_dict.values())
    mean_skld = np.mean(list(skld_dict.values()))
    # print(f'min_skld:{min_skld}, mean_skld:{mean_skld}')
    '''⬇️cannot/shouldn't merge'''
    # if min_skld > t_merge_fac * mean_skld:
    #     return Priors, Mu, Sigma,C_mat
    if cannot_merge_link != -1 and \
            ((ind_one == largst_comp or ind_two == nbComp - 1) or
             (ind_one == nbComp - 1 or  ind_two == largst_comp)):
        return Priors, Mu, Sigma,C_mat
    '''⬆️cannot/shouldn't merge'''
    print("Merging")
    tau_one, tau_two = Priors[0, ind_one], Priors[0, ind_two]
    tau_merged = tau_one + tau_two

    # c_mat_one, c_mat_two = C_mat[ind_one,0], C_mat[ind_two,0]
    c_mat_merged = 1

    f_one, f_two = tau_one / tau_merged, tau_two / tau_merged
    mu_one, mu_two = Mu[:, ind_one], Mu[:, ind_two]
    mu_merged = f_one * mu_one + f_two * mu_two
    sig_one, sig_two = Sigma[:, :, ind_one], Sigma[:, :, ind_two]
    sig_merged = f_one * sig_one + f_two * sig_two + \
                 f_one * f_two * (mu_one - mu_two).reshape(-1,1) @ (mu_one - mu_two).reshape(-1,1).T

    Priors[0, ind_one] = copy.deepcopy(tau_merged)
    Mu[:, ind_one] = copy.deepcopy(mu_merged)
    Sigma[:,:,ind_one] = copy.deepcopy(sig_merged)
    C_mat[ind_one,:] = copy.deepcopy(c_mat_merged)

    Priors = np.delete(Priors, [ind_two], axis=1)
    Mu = np.delete(Mu,[ind_two], axis = 1)
    # Assume ind_two belongs to axis 0.
    Sigma = np.delete(Sigma.T, [ind_two], axis=0)
    Sigma = Sigma.T
    C_mat = np.delete(C_mat, [ind_two], axis=0)

    return Priors, Mu, Sigma,C_mat

def _bic_func(_data_batch, Priors_in, Mu_in, Sigma_in):
    nb_states = Priors_in.shape[1]
    Post_pr_lst = []
    for m in range(Priors_in.shape[1]):
        this_post_pr = Priors_in[0, m].reshape(1) * gaussPDF_Func(_data_batch, Mu_in[:, m],
                                                                  Sigma_in[:, :, m]) + sys.float_info.min
        Post_pr_lst.append(this_post_pr)
    Post_pr = np.array(Post_pr_lst).reshape(Priors_in.shape[1], _data_batch.shape[1])
    psi = np.sum(Post_pr, axis = 0)
    log_like_for_batch = np.sum(np.log(psi))

    dimension = Mu_in.shape[0]
    M  = nb_states * (dimension + 1)*(dimension + 2) /2 -1
    _bic = -log_like_for_batch + np.log(_data_batch.shape[1]) * M/2
    return _bic

def fit_batch(_batch, max_nbStates):
    _all_bic = []
    _all_nbStates = []
    for nb_states in range(2, max_nbStates):
        _all_nbStates.append(nb_states)
        Priors_init, Mu_init, Sigma_init = EM_Init_Func(_batch, nb_states, True)
        em_Priors, em_Mu, em_Sigma = EM_Func(_batch,Priors_init, Mu_init, Sigma_init)
        this_bic = _bic_func(_batch,em_Priors, em_Mu, em_Sigma )
        _all_bic.append(this_bic)
    _all_bic = np.nan_to_num(_all_bic, nan=sys.float_info.max)
    best_nbstate = _all_nbStates[np.argmin(_all_bic)]
    return best_nbstate

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def update_policy_one(_batch_prev_norm, max_nbStates,lrn_rate, old_prior, old_mu, old_sigma,
                     new_prior, new_mu, new_sigma):

    for nb_com in range(new_prior.shape[1]):
        new_prior[0, nb_com] = lrn_rate * new_prior[0, nb_com]
    for nb_com in range(old_prior.shape[1]):
        old_prior[0, nb_com] = (1 - lrn_rate) * old_prior[0, nb_com]

    all_skld = []
    old_gmm_nb, new_gmm_nb = old_prior.shape[1], new_prior.shape[1]
    for ind_i in range(old_gmm_nb):
        this_old_skld = []
        for ind_j in range(new_gmm_nb):
            sig_A, sig_B = old_sigma[:, :, ind_i], new_sigma[:, :, ind_j]
            mu_A, mu_B = old_mu[:, ind_i], new_mu[:, ind_j]
            this_skld = skld_func(sig_A, sig_B, mu_A, mu_B)
            this_old_skld.append(this_skld)
        all_skld.append(this_old_skld)
    all_skld_arr = np.array(all_skld).reshape(old_gmm_nb, new_gmm_nb)
    '''⬇️maintain the maximum number of gaussians'''
    while (old_gmm_nb + new_gmm_nb ) > max_nbStates:
        pass
        (ind_one, ind_two) = np.unravel_index(np.argmin(all_skld_arr, axis=None), all_skld_arr.shape)
        all_skld_arr[ind_one, ind_two] = sys.float_info.max

        tau_one, tau_two = old_prior[0, ind_one], new_prior[0, ind_two]
        tau_sum = tau_one + tau_two
        f_one, f_two = tau_one / tau_sum, tau_two / tau_sum
        tau_merged = f_one + f_two

        mu_one, mu_two = old_mu[:, ind_one], new_mu[:, ind_two]
        mu_merged = f_one * mu_one + f_two * mu_two
        sig_one, sig_two = old_sigma[:, :, ind_one], new_sigma[:, :, ind_two]
        sig_merged = f_one * sig_one + f_two * sig_two + \
                     f_one * f_two * (mu_one - mu_two).reshape(-1,1) @ (mu_one - mu_two).reshape(-1,1).T

        old_prior[0, ind_one] = copy.deepcopy(tau_merged)
        old_mu[:, ind_one] = copy.deepcopy(mu_merged)
        old_sigma[:, :, ind_one] = copy.deepcopy(sig_merged)
        new_gmm_nb -=1
    if new_prior.shape[1] > 0:
        old_prior = np.hstack((old_prior, new_prior))
        old_mu = np.hstack((old_mu, new_mu))
        old_sigma = np.concatenate((old_sigma, new_sigma), axis=2)
    return old_prior, old_mu, old_sigma

def update_policy_two(old_sample_nb_N, batch_size, old_prior, old_mu, old_sigma,
                     new_prior, new_mu, new_sigma):
    t_merge = 1e3
    all_skld = []
    old_gmm_nb, new_gmm_nb = old_prior.shape[1], new_prior.shape[1]
    for ind_i in range(old_gmm_nb):
        this_old_skld = []
        for ind_j in range(new_gmm_nb):
            sig_A, sig_B = old_sigma[:, :, ind_i], new_sigma[:, :, ind_j]
            mu_A, mu_B = old_mu[:, ind_i], new_mu[:, ind_j]
            this_skld = skld_func(sig_A, sig_B, mu_A, mu_B)
            this_old_skld.append(this_skld)
        all_skld.append(this_old_skld)
    all_skld_arr = np.array(all_skld).reshape(old_gmm_nb, new_gmm_nb)
    '''⬇️maintain the maximum number of gaussians'''
    while all_skld_arr.min()  < t_merge:
        pass
        (ind_one, ind_two) = np.unravel_index(np.argmin(all_skld_arr, axis=None), all_skld_arr.shape)
        all_skld_arr[ind_one, ind_two] = sys.float_info.max

        mu_one, mu_two = old_mu[:, ind_one], new_mu[:, ind_two]
        tau_one, tau_two = old_prior[0, ind_one], new_prior[0, ind_two]
        mu_merged = (old_sample_nb_N * tau_one* mu_one + batch_size * tau_two * mu_two) \
                    / (old_sample_nb_N * tau_one + batch_size * tau_two )
        tau_merged =(old_sample_nb_N * tau_one + batch_size * tau_two ) / (old_sample_nb_N + batch_size)

        sig_one, sig_two = old_sigma[:, :, ind_one], new_sigma[:, :, ind_two]

        mu_one_square = (mu_one).reshape(-1,1) @ (mu_one).reshape(-1,1).T
        mu_two_square = (mu_two).reshape(-1, 1) @ (mu_two).reshape(-1, 1).T

        sig_merged = (old_sample_nb_N * tau_one * sig_one + batch_size * tau_two * sig_two) \
                     / (old_sample_nb_N * tau_one + batch_size * tau_two ) + \
                     (old_sample_nb_N * tau_one* mu_one_square + batch_size * tau_two * mu_two_square) \
                     /(old_sample_nb_N * tau_one + batch_size * tau_two ) \
                     - (mu_merged).reshape(-1,1) @ (mu_merged).reshape(-1,1).T

        old_prior[0, ind_one] = copy.deepcopy(tau_merged)
        old_mu[:, ind_one] = copy.deepcopy(mu_merged)
        old_sigma[:, :, ind_one] = copy.deepcopy(sig_merged)
        new_gmm_nb -=1
    if new_prior.shape[1] > 0:
        old_prior = np.hstack((old_prior, new_prior))
        old_mu = np.hstack((old_mu, new_mu))
        old_sigma = np.concatenate((old_sigma, new_sigma), axis=2)
    return old_prior, old_mu, old_sigma

def merge_new_into_old(old_sample_nb_N, batch_size, _batch_prev_norm ,_batch_next_norm,
                       max_nbStates,lrn_rate,
                       old_prior, old_mu, old_sigma,
                       new_prior, new_mu, new_sigma):

    old_prior_one, old_mu_one, old_sigma_one = update_policy_one(_batch_prev_norm, max_nbStates,lrn_rate,
                                                                 old_prior, old_mu, old_sigma,
                                                                 new_prior, new_mu, new_sigma)

    # old_prior_two, old_mu_two, old_sigma_two= update_policy_two(old_sample_nb_N, batch_size, old_prior, old_mu, old_sigma,
    #                  new_prior, new_mu, new_sigma)

    # in_out_split = _batch_next_norm.shape[0] - 1
    # this_exp_y_norm_one, dummy_Gaus_weights_one = GMR_Func(old_prior_one, old_mu_one, old_sigma_one,
    #                                                _batch_next_norm[:in_out_split, :], in_out_split)
    # this_exp_y_norm_two, dummy_Gaus_weights_two = GMR_Func(old_prior_two, old_mu_two, old_sigma_two,
    #                                                _batch_next_norm[:in_out_split, :], in_out_split)

    return old_prior_one, old_mu_one, old_sigma_one

def online_init(train_norm, max_nbStates):
    best_nbstate = fit_batch(train_norm, max_nbStates)
    Priors_init, Mu_init, Sigma_init = EM_Init_Func(train_norm, best_nbstate, False)
    old_prior, old_mu, old_sigma = EM_Func(train_norm, Priors_init, Mu_init, Sigma_init)
    return old_prior, old_mu, old_sigma

def online_update(old_sample_size, batch_size, _batch_prev_norm, _batch_next_norm,
                  max_nbStates,lrn_rate,
                  old_prior, old_mu, old_sigma):
    best_nbstate = fit_batch(_batch_prev_norm, batch_size)
    Priors_init, Mu_init, Sigma_init = EM_Init_Func(_batch_prev_norm, best_nbstate, True)
    new_prior, new_mu, new_sigma = EM_Func(_batch_prev_norm, Priors_init, Mu_init, Sigma_init)
    old_prior, old_mu, old_sigma = merge_new_into_old(old_sample_size, batch_size,_batch_prev_norm,_batch_next_norm,
                                                      max_nbStates, lrn_rate,
                                                      old_prior, old_mu, old_sigma,
                                                      new_prior, new_mu, new_sigma)
    return old_prior, old_mu, old_sigma

def online_norm(_batch):
    _batch_norm = normalize(_batch, axis = 1, norm='l1')
    sum_y = np.sum(_batch[-1,:])
    return _batch_norm, sum_y

def online_ggmr(train_norm, Data_Test ,max_nbStates, lrn_rate):
    train_norm = np.delete(train_norm, -2, axis=0) # delete rc_y information
    Data_Test = np.delete(Data_Test, -2, axis=0)  # delete rc_y information

    old_prior, old_mu, old_sigma = online_init(train_norm, max_nbStates)

    nbVar = Data_Test.shape[0]
    in_out_split = nbVar - 1
    _look_back_batch_size = 5
    _predict_size = 5
    expData = np.array([])
    for t_stamp in range(0, Data_Test.shape[1], _predict_size):
        print(t_stamp)
        if t_stamp > _look_back_batch_size:
            _batch_prev_norm = Data_Test[:, t_stamp - _look_back_batch_size: t_stamp]
            _batch_next_norm = Data_Test[:, t_stamp: t_stamp + _predict_size]
            old_prior, old_mu, old_sigma = online_update(t_stamp, _look_back_batch_size,_batch_prev_norm,_batch_next_norm,
                                                         max_nbStates,lrn_rate,
                                                         old_prior, old_mu, old_sigma)
        _batch_norm = Data_Test[:, t_stamp: t_stamp + _predict_size]
        this_exp_y_norm, dummy_Gaus_weights = GMR_Func(old_prior, old_mu, old_sigma ,
                                                  _batch_norm[:in_out_split,:], in_out_split)
        expData = np.concatenate((expData, this_exp_y_norm.reshape(-1)))
    return expData


def ggmr_func(Priors, Mu, Sigma, Data_Test,SumPosterior, L_rate, T_sigma):
    nbStates = Priors.shape[1]
    max_nbStates = nbStates + 2
    t_split_fac = 2e10
    t_merge_fac = 4e-1
    C_mat = SumPosterior.T
    nbVar = Data_Test.shape[0]
    in_out_split = nbVar - 1
    expData = []
    for t_stamp in range(Data_Test.shape[1]):
        # print(t_stamp)
        this_exp_y, dummy_Gaus_weights = GMR_Func(Priors, Mu, Sigma, Data_Test[:in_out_split, t_stamp], in_out_split)
        expData.append(this_exp_y)
        [Priors, Mu, Sigma, C_mat] = ggmr_create_update_gaussian(Data_Test,Priors, Mu, Sigma, t_stamp, C_mat, L_rate, T_sigma)
        Priors, Mu, Sigma, C_mat, cannot_merge_link, largst_volume_ind = split_func(Priors, Mu, Sigma,C_mat,t_split_fac, t_stamp, max_nbStates)
        Priors, Mu, Sigma, C_mat = merge_func(Priors, Mu, Sigma,C_mat, t_merge_fac, cannot_merge_link, largst_volume_ind)
        Priors = Priors / np.sum(Priors)

    return expData



def hybrid_func(Priors, Mu, Sigma, Data_Test,SumPosterior, test_initial_time,
    center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd, L_rate,T_Sigma):
    C_mat = SumPosterior.T
    nbVar = Data_Test.shape[0]
    in_out_split = nbVar - 1
    expData = []

    rc_real_pre = RC_Prediction(abcd)
    for t in range(Data_Test.shape[1]):
        '''⬇️Updating rc load information'''
        target_time = t +test_initial_time
        u_arr = u_measured[:,target_time - rc_warming_step:target_time+1]
        rc_result = rc_real_pre.predict(u_arr)
        rc_result_norm = (rc_result - center_rc_y) / scale_rc_y
        Data_Test[-2,t] = rc_result_norm
        '''⬆️Updating rc load information'''
        this_exp_y, dummy_Gaus_weights = GMR_Func(Priors, Mu, Sigma, Data_Test[:in_out_split, t], in_out_split)
        expData.append(this_exp_y)
        [Priors, Mu, Sigma, C_mat] = hybrid_update_gaussian(Data_Test,Priors, Mu, Sigma, t, C_mat, L_rate, T_Sigma)

    return expData

class RC_Prediction():
    def __init__(self, pre_abcd):
        self.abcd = pre_abcd

    def predict(self, u_arr):
        y_model = np.zeros((u_arr.shape[1],))
        x_discrete = np.array([[0], [10], [22], [21], [23], [21]])
        for i in range(u_arr.shape[1]):
            y_model[i] = (self.abcd['c'] @ x_discrete + self.abcd['d'] @ u_arr[:, i])[0, 0]
            x_discrete = self.abcd['a'] @ x_discrete + (self.abcd['b'] @ u_arr[:, i]).reshape((6, 1))
        return y_model[-1]