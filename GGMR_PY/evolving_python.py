import matlab.engine
import numpy as np


def Evolving_LW_2(Priors, Mu, Sigma, Data_Test,SumPosterior,talk_to_rc, test_initial_time,
    center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd, L_rate):
    eng = matlab.engine.start_matlab()
    L1 = Data_Test.shape[0]
    nes = L1 - 1
    expData = []
    for t in range(1, Data_Test.shape[1]):
        cur_resut_GMR = eng.GMR(matlab.double(Priors.tolist()) , matlab.double(Mu.tolist()), matlab.double(Sigma.tolist()),
                                      matlab.double(Data_Test[:L1 - 1, t].tolist()),
                                      matlab.double(range(1, nes +  1)), matlab.double(L1))
        # print(eng.triarea(1, 5))

    Priors, Mu , Sigma, expData = 0, 0 ,0,0
    return [Priors, Mu, Sigma, expData]








