import matplotlib.pyplot as plt
import pandas as pd
import lmfit


def resid(params, d_Tz, Te_Ta, d_Te, Tz_Ta, ydata):
    c_z = params['c_z'].value
    d_t = params['d_t'].value
    r_ea = params['r_ea'].value
    c_e = params['c_e'].value
    r_za = params['r_za'].value

    y_model = c_z / d_t * d_Tz + Te_Ta / r_ea + c_e / d_t * d_Te + Tz_Ta / r_za
    return y_model - ydata


def para():
    params = lmfit.Parameters()
    params.add_many(('c_z', 1.6e5),
                    ('d_t', 120, False),
                    ('r_ea', 0.03),
                    ('c_e', 1e6),
                    ('r_za', 3))
    return params

def arg():

    input_df = pd.read_csv('./Case600_sim.csv', index_col=0, parse_dates=True)
    d_Tz = input_df['dTz'][:5040]
    Te_Ta = input_df['Te_Ta'][:5040]
    d_Te = input_df['dTe'][:5040]
    Tz_Ta = input_df['Tz_Ta'][:5040]
    ydata = input_df['y_heating'][:5040]
    return (d_Tz, Te_Ta, d_Te, Tz_Ta, ydata)

def plot(o1):
    input_df = pd.read_csv('./Case600_sim.csv', index_col=0, parse_dates=True)
    yn = input_df['y_heating'][:5040]
    max_heating = max(yn)

    plt.plot([0, max_heating], [0, max_heating], '-', label='data')
    plt.plot(yn, yn + o1.residual, 'o', label='modeled')
    plt.legend()
    plt.show()

if __name__ == '__main__' :
    para = para()
    arg = arg()
    o1 = lmfit.minimize(resid, para, args=arg, method='leastsq')
    # print("# Fit using leastsq:")
    lmfit.report_fit(o1)
    plot(o1)
