from multiprocessing import freeze_support
from RC_training import _3_rc_py_main as RC_package
from RC_training import _2_pyswarm as RC_utils
from RC_prediction import call_rc_function

if __name__ == '__main__':
    freeze_support()

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
    rc_obj.train(_test_start= 4033, _n_particle=500, _iters=50) #training
    # rc_obj.train(_test_start=4033) #use trained RsCs get predicted
    # # rc_obj.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5, _n_particle=1000, _iters=20)
    # # rc_obj.train(_test_start=4033, _state_num=4, _input_num=8, _para_nums=19, _case_nbr=5)
    # # rc_obj.train(_test_start=4033, _state_num=5,_case_nbr=7, _n_particle=1000, _iters=20)
    # rc_obj.train(_test_start=4033, _state_num=5, _case_nbr=7)

    '''
    system rc performance visualization
    '''
    # RC_utils.comparison_performance()

