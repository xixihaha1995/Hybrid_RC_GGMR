from RC import _3_rc_py_main as RC_package

'''
RC prediction
input: time step index
output: predicted heating(positive) or cooling(load) for that time step index

Usage:
rc_obj = RC_package.RC_Class() # call this once

for time_idx in [66, 70]:
 print(rc_obj.predict(_time_idx = time_idx))
'''

# rc_obj = RC_package.RC_Class() # call this once
#
# for time_idx in [66, 70]:
#  print(rc_obj.predict(_time_idx = time_idx))

'''
RC training
RC system, with sink node(Default)
rc_obj = RC_package.RC_Class()
rc_obj.train(_test_start= 4033)

RC system, without sink node
rc_obj = RC_package.RC_Class()
rc_obj.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5, _n_particle=2, _iters=2)
'''
rc_obj = RC_package.RC_Class()
rc_obj.train(_test_start= 4033, _n_particle=2, _iters=2)
# rc_obj.train(_test_start= 4033, _state_num=4, _input_num=8,_para_nums=19,_case_nbr=5, _n_particle=1000, _iters=150)



