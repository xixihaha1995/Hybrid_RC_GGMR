import os, _utils
import scipy.io as sio
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','_em_init_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

train_norm = mat_contents['train_norm']
nbStates = mat_contents['nbStates'][0,0]

init_Priors, init_Mu, init_Sigma = _utils.EM_Init_Func(train_norm, nbStates)
em_Priors, em_Mu, em_Sigma  = _utils.EM_Func(train_norm, init_Priors, init_Mu, init_Sigma)
print(em_Priors, em_Mu, em_Sigma)