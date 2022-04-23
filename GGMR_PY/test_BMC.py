import os, _utils, matlab.engine
import scipy.io as sio
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','_bmc_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

Data_Test = mat_contents['Data_Test']
nbVar = mat_contents['L1'][0,0]
t = mat_contents['t'][0,0]
Priors = mat_contents['Priors']
Mu = mat_contents['Mu']
Sigma = mat_contents['Sigma']

print(_utils.BMC_Func(Data_Test[:,t], Priors, Mu, Sigma))