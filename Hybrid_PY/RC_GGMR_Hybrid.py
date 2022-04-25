'''
TODO: Load inputs
GMR
GGMR
Hybrid
'''
import _0_generic_utils as general_tools, _1_gmr_ggmr_hybrid_utils as gaussian_tools
import matplotlib.pyplot as plt, os, scipy.io as sio
nbStates = 15


All_Variables_obj = general_tools.switch_case(0)
All_Variables = All_Variables_obj.astype('float64')
total_length = All_Variables.shape[1]
training_length = 4032
test_initial_time = training_length -1
rc_warming_step = 14
testing_length = total_length - training_length
# testing_length = 1000
talk_to_rc  = 0

nbVarAll = All_Variables.shape[0]
nbVarInput = nbVarAll - 1

train, test, train_norm, test_norm = general_tools.split_train_test_norm(
    nbVarAll, All_Variables,training_length, testing_length)

center_rc_y, scale_rc_y = train[-2,:].mean(), train[-2,:].std()
# init_Priors, init_Mu, init_Sigma = gaussian_tools.EM_Init_Func(train_norm, nbStates)
# em_Priors_py, em_Mu_py, em_Sigma_py  = gaussian_tools.EM_Func(train_norm, init_Priors, init_Mu, init_Sigma)

script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','_skip_em.mat')
mat_contents = sio.loadmat(mat_fname)
em_Priors = mat_contents['rs_Priors']
em_Mu = mat_contents['rs_Mu']
em_Sigma = mat_contents['rs_Sigma']
test_norm = mat_contents['test_norm']

gmr_norm, gmr_beta = gaussian_tools.GMR_Func(em_Priors,em_Mu, em_Sigma, test_norm[:nbVarInput,:], nbVarInput)
center_y, scale_y = train[-1,:].mean(), train[-1,:].std()
gmr_predict = gmr_norm * scale_y + center_y
gmr_predict = gmr_predict.reshape(-1)

y_test = test[-1,:]
rc_y = test[-2,:]
rmse_gmr = (sum((y_test - gmr_predict) ** 2) / len(y_test)) ** (1 / 2)
fig, ax = plt.subplots(2)

ax[0].plot(y_test, label = "Measured")
ax[0].plot(rc_y, label = "RC")
ax[0].plot(gmr_predict, label = "GMR")
ax[0].legend()
plt.show()

pass

