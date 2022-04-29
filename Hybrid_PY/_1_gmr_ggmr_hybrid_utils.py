import sys, numpy as np, copy
from sklearn.cluster import KMeans

def gaussPDF_Func(Data, Mu, Sigma):
    if Data.ndim == 1:
        nbVar, nbData = Data.shape[0], 1
    else:
        nbVar, nbData = Data.shape
    Data = Data.T - np.tile(Mu.T, [nbData, 1])
    prob = np.sum(Data @ np.linalg.inv(Sigma) * Data, axis=1)
    prob = np.exp(-0.5 * prob )/ np.sqrt((2 * np.pi) ** nbVar * (abs(np.linalg.det(Sigma)) + sys.float_info.min) )
    return prob

def EM_Init_Func(Data, nbStates):
    Data_tran = Data.T
    nbVar = Data_tran.shape[1]
    minc = np.min(Data_tran, axis = 0)
    maxc = np.max(Data_tran, axis=0)
    all_var_ran = []
    for idx_var in range(nbVar):
        step = (maxc[idx_var] - minc[idx_var]) / (nbStates)
        this_var_ran = np.arange(minc[idx_var], maxc[idx_var] - 1e-5, step)
        all_var_ran.append(this_var_ran)
    all_var_cen = np.array(all_var_ran).T
    kmeans = KMeans(n_clusters=nbStates, init=all_var_cen,random_state=0).fit(Data_tran)
    # kmeans = KMeans(n_clusters=nbStates, algorithm="elkan").fit(Data_tran)
    Mu = kmeans.cluster_centers_.T
    Priors_lst = []
    Sigma_lst = []
    for cluster_idx in range(nbStates):
        Priors_lst.append(Data_tran[np.where(kmeans.labels_ == cluster_idx)].shape[0])
        this_cluster_samps = Data_tran[np.where(kmeans.labels_ == cluster_idx)].T
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
        Pix_nan = Pix_tmp / np.tile(np.sum(Pix_tmp, axis=1).reshape(-1,1),[1, nbStates])

        Pix = np.nan_to_num(Pix_nan, nan=0)
        # compute the cumulated posterior probability
        E = np.sum(Pix, axis = 0).reshape(1,-1)
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
        loglik = np.log(F).mean()
        # print(abs((loglik/loglik_old)-1))

        if abs((loglik / loglik_old) - 1) < loglik_threshold:
            break
        loglik_old = loglik

    Sigma[:, :, :] += 1e-5 * np.identity(nbVar).reshape(nbVar,nbVar,-1)
    return Priors, Mu, Sigma

def GMR_Func(Priors, Mu, Sigma, input_x, in_out_split):
    nbVar = Mu.shape[0]
    nbVarInput = nbVar - 1
    input_x = input_x.reshape(nbVarInput,-1)
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
        this_post_pr = Priors_in[0, m].reshape(1) @ gaussPDF_Func(Data_in, Mu_in[:,m], Sigma_in[:,:,m])
        Post_pr_lst.append(this_post_pr)
    Post_pr = np.array(Post_pr_lst).reshape(-1,1)
    m_best = np.argmax(Post_pr)
    return m_best, Post_pr

def Mahal_dis_Func(Data,Mu,Cov):
    pass
    # Md = sqrt(abs((Data - Mu)'*inv(Cov)*(Data-Mu)));
    Md = np.sqrt(abs((Data - Mu).T @ np.linalg.inv(Cov) @ (Data - Mu)))
    return Md

def ggmr_update_gaussian(Data_Test,Priors, Mu, Sigma, t, C_mat, L_rate, T_sigma):
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
        Update all Gaussians
        '''
        # print("Updating")
        for nb_com in range(Post_pr.shape[0]):
            q_j = Post_pr[nb_com, 0] / np.sum(Post_pr, axis= 0 )
            C_mat[nb_com, 0] += q_j
            tau_j =( (1 - L_rate) * Priors[0,nb_com] + L_rate * q_j)[0]
            Priors[0,nb_com] = min(tau_j, tau_min_thres)
            eta_j = q_j * ((1 - L_rate) / C_mat[nb_com, 0] + L_rate)
            eta_j = eta_j[0]
            Mu[:, nb_com] = (1 - eta_j) * Mu[:, nb_com] + eta_j * Data_Test[:,t]
            x_min_mu = (Data_Test[:, t] - Mu[:, nb_com]).reshape(-1, 1)
            Sigma[:,:,nb_com] = (1 - eta_j) * Sigma[:,:,nb_com] + eta_j * x_min_mu @ x_min_mu.T

    return [Priors, Mu, Sigma, C_mat]

def split_func(Priors, Mu, Sigma, t_split, time_stam):
    split_factor = 8e-1
    # can volume be negative
    cannot_merge_link = -1
    if min(np.linalg.det(Sigma.T)) < 0:
        print("Sigma volumes can be negative")
    if max(np.linalg.det(Sigma.T)) < t_split:
        return Priors, Mu, Sigma,cannot_merge_link
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
    mu_one = mu_prev + delta_eigen_vect
    mu_two = mu_prev - delta_eigen_vect
    sigma_one = sigma_two = Sigma.T[largst_comp,:,:] - delta_eigen_vect @ delta_eigen_vect.T

    Priors[0, largst_comp] = copy.deepcopy(tau_one)
    Mu[:, largst_comp] = copy.deepcopy(mu_one)
    Sigma[:,:,largst_comp] = copy.deepcopy(sigma_one)

    Priors = np.hstack((Priors, tau_two))
    Mu = np.hstack((Mu, mu_two.reshape(-1,1)))
    Sigma = np.vstack((Sigma.T,sigma_two[None]))
    Sigma = Sigma.T
    return Priors, Mu, Sigma, cannot_merge_link, largst_comp

def skld_func(sig_one, sig_two, mu_one, mu_two):
    D = sig_one.shape[0]
    kld_one = 1/2 * (np.log(np.linalg.det(sig_one) / np.linalg.det(sig_two)) \
              + np.trace(np.linalg.inv(sig_two) @ sig_one) \
              + (mu_two - mu_one).reshape(-1,1).T @ np.linalg.inv(sig_one) @ (mu_two - mu_one).reshape(-1,1) - D)
    kld_two =1/2 * (np.log(np.linalg.det(sig_two) / np.linalg.det(sig_one)) \
              + np.trace(np.linalg.inv(sig_one) @ sig_two) \
              + (mu_one - mu_two).reshape(-1,1).T @ np.linalg.inv(sig_two) @ (mu_one - mu_two).reshape(-1,1) - D)
    skld = 1/2 * (kld_one + kld_two)
    return skld

def merge_func(Priors, Mu, Sigma,t_merge, cannot_merge_link, largst_comp):
    skld_2d = []
    nbComp = Sigma.shape[0]
    for ind_i in range(nbComp):
        skld_level = []
        for ind_j in range(ind_i + 1, nbComp):
            sig_A, sig_B = Sigma[:,:,ind_i],Sigma[:,:,ind_j]
            mu_A, mu_B = Mu[:,ind_i], Mu[:,ind_j]
            this_skld = skld_func(sig_A, sig_B, mu_A, mu_B)
            skld_level.append(this_skld)
        skld_2d.append(skld_level)
    skld_2d = abs(np.array(skld_2d))
    (ind_one, ind_two) = np.unravel_index(skld_2d.argmin(), skld_2d.shape)
    '''⬇️cannot/shouldn't merge'''
    if cannot_merge_link != -1 and \
            ((ind_one == largst_comp and ind_two == nbComp - 1) or
             (ind_one == nbComp - 1 and largst_comp)):
        return Mu, Sigma

    if skld_2d.min() > t_merge:
        return Mu, Sigma
    '''⬆️cannot/shouldn't merge'''
    tau_one, tau_two = Priors[0, ind_one], Priors[0, ind_two]
    tau_merged = tau_one + tau_two
    f_one, f_two = tau_one / tau_merged, tau_two / tau_merged
    mu_one, mu_two = Mu[:, ind_one], Mu[:, ind_two]
    mu_merged = f_one * mu_one + f_two * mu_two
    sig_one, sig_two = Sigma[ind_one, :, :], Sigma[ind_two, :, :]
    sig_merged = f_one * sig_one + f_two * sig_two + \
                 f_one * f_two * (mu_one - mu_two).reshape(-1,1) @ (mu_one - mu_two).reshape(-1,1).T

    Priors[0, largst_comp] = copy.deepcopy(tau_merged)
    Mu[:, largst_comp] = copy.deepcopy(mu_merged)
    Sigma[:,:,largst_comp] = copy.deepcopy(sig_merged)

    

    return Priors, Mu, Sigma



def ggmr_func(Priors, Mu, Sigma, Data_Test,SumPosterior, L_rate, T_sigma):
    t_split = 5e-1
    t_merge = 5
    C_mat = SumPosterior.T
    nbVar = Data_Test.shape[0]
    in_out_split = nbVar - 1
    expData = []
    for t_stamp in range(Data_Test.shape[1]):
        this_exp_y, dummy_Gaus_weights = GMR_Func(Priors, Mu, Sigma, Data_Test[:in_out_split, t_stamp], in_out_split)
        expData.append(this_exp_y)
        [Priors, Mu, Sigma, C_mat] = ggmr_update_gaussian(Data_Test,Priors, Mu, Sigma, t_stamp, C_mat, L_rate, T_sigma)
        Priors, Mu, Sigma,cannot_merge_link, largst_volume_ind = split_func(Priors, Mu, Sigma,t_split, t_stamp)
        Mu, Sigma = merge_func(Priors, Mu, Sigma,t_merge, cannot_merge_link, largst_volume_ind)
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
        [Priors, Mu, Sigma, C_mat] = ggmr_update_gaussian(Data_Test,Priors, Mu, Sigma, t, C_mat, L_rate, T_Sigma)

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