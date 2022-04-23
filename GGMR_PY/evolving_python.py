import matlab.engine, _utils
import numpy as np


def Evolving_LW_2(Priors, Mu, Sigma, Data_Test,SumPosterior,talk_to_rc, test_initial_time,
    center_rc_y, scale_rc_y,u_measured, rc_warming_step,abcd, L_rate):
    eng = matlab.engine.start_matlab()
    nbVar = Data_Test.shape[0]
    in_out_split = nbVar - 1
    expData = []
    for t in range(1, Data_Test.shape[1]):
        _utils.GMR_Func(Priors, Mu, Sigma, Data_Test[:in_out_split, t], in_out_split)
        
    Priors, Mu , Sigma, expData = 0, 0 ,0,0
    return [Priors, Mu, Sigma, expData]








