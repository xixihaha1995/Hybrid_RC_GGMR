import os, Hybrid_PY._1_gmr_ggmr_hybrid_utils as gaussian_tools
import scipy.io as sio
script_dir = os.path.dirname(__file__)
mat_fname = os.path.join(script_dir,'inputs','evolving_inputs.mat')
mat_contents = sio.loadmat(mat_fname)

rs_Priors_ggmr = mat_contents['rs_Priors_ggmr']
rs_Mu_ggmr = mat_contents['rs_Mu_ggmr']
rs_Sigma_ggmr = mat_contents['rs_Sigma_ggmr']
test_norm_ggmr = mat_contents['test_norm_ggmr']
sum_beta_rs_ggmr = mat_contents['sum_beta_rs_ggmr']
ggmr_talk_rc = mat_contents['ggmr_talk_rc']
test_initial_time = mat_contents['test_initial_time']
center_rc_y = mat_contents['center_rc_y']
scale_rc_y = mat_contents['scale_rc_y']
u_measured = mat_contents['u_measured']
rc_warming_step = mat_contents['rc_warming_step']
abcd = mat_contents['abcd']
L_rate = mat_contents['L_rate']
expData = gaussian_tools.Evolving_LW_2_Func(rs_Priors_ggmr, rs_Mu_ggmr, rs_Sigma_ggmr,
                                                             test_norm_ggmr,sum_beta_rs_ggmr, ggmr_talk_rc,
                                                             test_initial_time, center_rc_y, scale_rc_y,
                                                             u_measured, rc_warming_step,abcd,L_rate)
print("dummy")