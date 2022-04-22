import matlab.engine

def Evolving_LW_2(Priors, Mu, Sigma, Data_Test,SumPosterior,talk_to_rc, test_initial_time,
    center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd, L_rate):
    eng = matlab.engine.start_matlab()
    L1 = Data_Test.shape[0]
    print(f'I am here: {L1}')
    print(type(Priors))
    print(Priors)
    # for t in range(2, Data_Test.shape[1]):

        # [expData(t, 1), cof1] = eng.GMR(Priors, Mu, Sigma, Data_Test(1:L1 - 1, t), [1: nes], [nes + 1: nbVar])

    Priors, Mu , Sigma, expData = 0, 0 ,0,0
    return [Priors, Mu, Sigma, expData]

# res = Evolving_LW_2(rs_Priors_ggmr, rs_Mu_ggmr, rs_Sigma_ggmr, test_norm_ggmr,
#                   sum_beta_rs_ggmr, ggmr_talk_rc, test_initial_time, center_rc_y, scale_rc_y,
#                   u_measured, rc_warming_step,abcd,L_rate)






