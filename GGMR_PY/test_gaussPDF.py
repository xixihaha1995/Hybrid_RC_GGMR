import os, gaussPDF
import scipy.io as sio
import matlab.engine
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','_gaussPDF_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

Data = mat_contents['Data']
Mu = mat_contents['Mu']
Sigma = mat_contents['Sigma']

print(gaussPDF.gaussPDF_Func(Data, Mu, Sigma))