from multiprocessing import freeze_support
from RC_training import _3_rc_py_main as RC_package
from RC_training import _2_pyswarm as RC_utils
from RC_prediction import call_rc_function

if __name__ == '__main__':
    freeze_support()
    '''
    #RC prediction_with_warming
    #input: u_measured, shape(input_dimension, warming_up_steps = 15)
    #output: predicted heating(positive) or cooling(negative)
    '''
    # target_time_idx = 782
    # warming_up_steps = 15
    # u_measured = RC_utils.warming_input_demo(time_idx = target_time_idx, seg_length= warming_up_steps)
    # #The above codes is to show the expected input array
    #
    # print(f'u_measured shape:{u_measured.shape}')
    # # print(f'u_measured:{u_measured}')
    # rc_function = call_rc_function.RC_Prediction()
    # print(f'RC prediction with pre-calcualted abcd:{rc_function.predict(u_measured)} Watts load')

    '''
    #RC training
    #RC system, with sink node(Default)
    rc_obj = RC_package.RC_Class()
    rc_obj.train(_test_start= 4033)
    
    #RC system, without sink node
    rc_obj = RC_package.RC_Class()
    rc_obj.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5, _n_particle=2, _iters=2)
    '''
    rc_obj = RC_package.RC_Class()
    # # rc_obj.train(_test_start= 4033, _n_particle=1000, _iters=150) #training
    # rc_obj.train(_test_start=4033) #use trained RsCs get predicted
    rc_obj.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5, _n_particle=1000, _iters=20)
    # rc_obj.train(_test_start=4033, _state_num=4, _input_num=8, _para_nums=19, _case_nbr=5)

