import os, Hybrid_PY._1_gmr_ggmr_hybrid_utils as gaussian_tools
import scipy.io as sio
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'..','inputs','_gaussPDF_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

Data = mat_contents['Data']
Mu = mat_contents['Mu']
Sigma = mat_contents['Sigma']

print(gaussian_tools.gaussPDF_Func(Data, Mu, Sigma))