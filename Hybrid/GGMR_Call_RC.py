print("Hello")
import sys
from RC_Package.RC_training import _2_pyswarm as RC_utils
from RC_Package.RC_prediction import call_rc_function
#
# def recv_send(target_time_idx):
#     warming_up_steps = 15
#     u_measured = RC_utils.warming_input_demo(time_idx = target_time_idx, seg_length= warming_up_steps)
#     rc_function = call_rc_function.RC_Prediction()
#     rc_y = rc_function.predict(u_measured)
#     return rc_y
#
# target_time_idx = int(sys.argv[1])
# rc_y = recv_send(target_time_idx)
# print(f'RC prediction with pre-calcualted abcd:{rc_y} Watts load')