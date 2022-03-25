from RC import _3_rc_py_main as RC_package

'''
(Default)
RC system, sink as state variable
'''
# RC_package.train(_test_start= 4033)
'''
RC system, slab as boundary temperature
-a 300 0 2016 4 8 19 2 2 5 0
'''
RC_package.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5)



