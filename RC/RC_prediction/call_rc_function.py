import json
import numpy as np, os

class RC_Prediction():
    def __init__(self):
        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
        abcd_absolute = os.path.join(script_dir, 'abcd.json')
        self.abcd = json.load(open(abcd_absolute))

    def predict(self, u_arr):
        y_model = np.zeros((u_arr.shape[1],))
        x_discrete = np.array([[0], [10], [22], [21], [23], [21]])
        for i in range(u_arr.shape[1]):
            y_model[i] = (self.abcd['c'] @ x_discrete + self.abcd['d'] @ u_arr[:, i])[0, 0]
            x_discrete = self.abcd['a'] @ x_discrete + (self.abcd['b'] @ u_arr[:, i]).reshape((6, 1))
        return y_model[-1]