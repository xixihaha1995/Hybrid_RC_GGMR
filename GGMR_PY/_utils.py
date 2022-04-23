import sys, numpy as np
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