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
    y, beta = 0, 0
    return [y, beta]
