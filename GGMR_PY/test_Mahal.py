import os, _utils, matlab.engine
import scipy.io as sio
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','_mahal_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

Data_Test = mat_contents['Data_Test']
nbVar = mat_contents['L1'][0,0]
t = mat_contents['t'][0,0] - 1
m = mat_contents['m'][0,0] - 1
Mu = mat_contents['Mu']
Sigma = mat_contents['Sigma']

# com_MD(m,1)= Mahal_dis(Data_Test(1:L1,t),Mu(:,m),Sigma(:,:,m))

print(_utils.Mahal_dis_Func(Data_Test[:,t], Mu[:,m], Sigma[:,:,m]))