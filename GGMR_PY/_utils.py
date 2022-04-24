import sys, numpy as np
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

def GMR_Func(Priors, Mu, Sigma, input_x, in_out_split):
    input_x = input_x.reshape(-1,1)

    if input_x.ndim == 1:
        temp, nbData = input_x.shape[0], 1
    else:
        [temp, nbData] = input_x.shape
    nbVar = Mu.shape[0]
    nbStates = Sigma.shape[2]

    Px = []
    for i in range(nbStates):
        this_Pxi = Priors[0, i] * gaussPDF_Func(input_x, Mu[:in_out_split,i], Sigma[:in_out_split, :in_out_split, i])
        Px.append(this_Pxi)
    Px_reshape = np.array(Px).T
    beta = Px_reshape / np.tile(np.sum(Px_reshape, axis= 1) + sys.float_info.min,[1, nbStates])

    y_temp_lst = []
    for j in range(nbStates):
        this_y_tmp = np.tile(Mu[in_out_split:, j], [1, nbData]) + Sigma[in_out_split:,:in_out_split, j] \
                     @ np.linalg.inv(Sigma[:in_out_split,:in_out_split, j]) @ \
                     (input_x - np.tile(Mu[:in_out_split, j].reshape(-1,1),[1, nbData]))
        y_temp_lst.append(this_y_tmp)

    y_tmp = np.array(y_temp_lst).reshape(1,1,-1)
    beta_tmp = beta.reshape(1,1,-1)
    y_tmp2 = np.tile(beta_tmp,[nbVar - in_out_split, 1,1]) * y_tmp
    y = np.sum(y_tmp2, axis=2)

    # % % Compute expected covariance matrices Sigma_y, given input x
    # % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    # for j=1:nbStates
    # Sigma_y_tmp(:,:, 1, j) = Sigma(out, out, j) - (Sigma(out, in, j) * inv(Sigma( in, in, j))*Sigma( in, out, j));
    # end
    # beta_tmp = reshape(beta, [1 1 size(beta)]);
    # Sigma_y_tmp2 = repmat(beta_tmp. * beta_tmp, [length(out) length(out) 1 1]). * repmat(Sigma_y_tmp, [1 1 nbData 1]);
    # Sigma_y = sum(Sigma_y_tmp2, 4);
    # Sigma_y;

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

def EM_Init_Func(Data, nbStates):
    Data_tran = Data.T
    nbVar = Data_tran.shape[1]
    minc = np.min(Data_tran, axis = 1)
    maxc = np.max(Data_tran, axis=1)
    all_var_ran = []
    for idx_var in range(nbVar):
        step = (maxc[idx_var] - minc[idx_var]) / (nbStates)
        this_var_ran = np.arange(minc[idx_var], maxc[idx_var], step)
        all_var_ran.append(this_var_ran)
    all_var_cen = np.array(all_var_ran).T
    kmeans = KMeans(n_clusters=nbStates, init=all_var_cen,random_state=0).fit(Data_tran)
    Mu = kmeans.cluster_centers_.T
    Priors_lst = []
    Sigma_lst = []
    for cluster_idx in range(nbStates):
        Priors_lst.append(Data_tran[np.where(kmeans.labels_ == cluster_idx)].shape[0])
        this_cluster_samps = Data_tran[np.where(kmeans.labels_ == cluster_idx)]
        this_cluster_sigma = np.cov(this_cluster_samps.T) + 1e-5*np.identity(nbVar)
        Sigma_lst.append(this_cluster_sigma)
    Priors = np.array(Priors_lst) / np.sum(Priors_lst).reshape(1,-1)
    Sigma = np.array(Sigma_lst).reshape(nbVar,nbVar,-1)
    return Priors, Mu, Sigma

def EM_Func(Data, Priors0, Mu0, Sigma0):
    # for i=1:nbStates
    #     Sigma(:,:, i) = Sigma(:,:, i) + 1E-5. * diag(ones(nbVar, 1))
    loglik_threshold = 1e-10
    (nbVar, nbData) = Data.shape
    nbStates = Sigma0.shape[2]
    loglik_old = - sys.float_info.max
    nbStep = 0
    Mu = Mu0
    Sigma = Sigma0
    Priors = Priors0
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
        F = Pxi @ Priors.T
        loglik = np.log(F).mean()

        if abs((loglik / loglik_old) - 1) < loglik_threshold:
            break
        loglik_old = loglik

    Sigma[:, :, :] += 1e-5 * np.identity(nbVar).reshape(nbVar,nbVar,-1)




    return Priors, Mu, Sigma, Pix