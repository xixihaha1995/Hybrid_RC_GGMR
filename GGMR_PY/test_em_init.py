import os, _utils
import scipy.io as sio
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','_em_init_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

train_norm = mat_contents['train_norm']
nbStates = mat_contents['nbStates'][0,0]

print(_utils.EM_Init_Func(train_norm, nbStates))