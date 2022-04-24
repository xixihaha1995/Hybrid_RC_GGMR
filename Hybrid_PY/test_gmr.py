import os, _1_gmr_ggmr_hybrid_utils, matlab.engine
import scipy.io as sio
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','_gmr_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

Priors = mat_contents['Priors']
Mu = mat_contents['Mu']
Sigma = mat_contents['Sigma']
Data_Test = mat_contents['Data_Test']
nbVar = mat_contents['L1'][0,0]
t = mat_contents['t'][0,0]
in_out_split = nbVar - 1


print(_utils.GMR_Func(Priors, Mu, Sigma,Data_Test[:in_out_split,t], in_out_split))