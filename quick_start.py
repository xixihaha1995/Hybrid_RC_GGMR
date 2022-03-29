from multiprocessing import freeze_support
from RC import _3_rc_py_main as RC_package
from RC import _2_pyswarm as RC_utils

if __name__ == '__main__':
    freeze_support()
    '''
    #RC prediction_with_warming
    #input: u_measured, shape(input_dimension, warming_up_steps = 15)
    #output: predicted heating(positive) or cooling(load) for last time step
    
    #RC prediction_from_predicted
    #input: target_time_idx
    #output: predicted heating(positive) or cooling(load) for target_time_idx
    '''
    # target_time_idx = 782
    # warming_up_steps = 15
    # u_measured = RC_utils.warming_input_demo(time_idx = target_time_idx, seg_length= warming_up_steps)
    #
    # rc_obj = RC_package.RC_Class(case_nbr=6)  # call this once
    # print( rc_obj.predict_with_warming(u_measured))
    # print( rc_obj.load_predicted(target_time_idx))

    '''
    #RC training
    #RC system, with sink node(Default)
    rc_obj = RC_package.RC_Class()
    rc_obj.train(_test_start= 4033)
    
    #RC system, without sink node
    rc_obj = RC_package.RC_Class()
    rc_obj.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5, _n_particle=2, _iters=2)
    '''
    # rc_obj = RC_package.RC_Class()
    # # rc_obj.train(_test_start= 4033, _n_particle=1000, _iters=150) #training
    # # rc_obj.train(_test_start=4033) #use trained RsCs get predicted
    # rc_obj.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5)

    '''
    Case 5, 6 Visualization
    '''
    RC_utils.comparison_performance()

