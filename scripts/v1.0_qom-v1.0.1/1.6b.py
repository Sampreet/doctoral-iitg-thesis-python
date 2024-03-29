# dependencies
import numpy as np
import os 
import sys

# qom modules
from qom.utils.loopers import run_loopers_in_parallel, wrap_looper
from qom.utils.solvers import get_func_quantum_correlation_measures

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.Bidirectional import Bi_00

# all parameters
params = {
    'looper': {
        'show_progress'     : True,
        'file_path_prefix'  : 'data/v1.0_qom-v1.0.1/1.6b',
        'X'                 : {
            'var'   : 'delta',
            'min'   : -0.02,
            'max'   : 0.02,
            'dim'   : 101
        },
        'Y'                 : {
            'var'   : 'lambda',
            'min'   : 0.0,
            'max'   : 0.1,
            'dim'   : 101
        }
    },
    'solver': {
        'show_progress' : False,
        'cache'         : True,
        'measure_codes' : ['sync_p'],
        'indices'       : [1, 3],
        'ode_method'    : 'vode',
        't_min'         : 0.0,
        't_max'         : 1000.0,
        't_dim'         : 10001,
        't_index_min'   : 9371,
        't_index_max'   : 10000
    },
    'system': {
        'A_l'           : 52.0,
        'Delta_0_sign'  : 1.0, 
        'delta'         : 0.01,
        'g_0s'          : [0.005, 0.005],
        'gammas'        : [0.005, 0.005],
        'kappas'        : [0.15, 0.15],
        'lambda'        : 0.0375,
        'n_ths'         : [0.0, 0.0],
        'omega_mL'      : 1.0
    },
    'plotter': {
        'type'              : 'contourf',
        'x_label'           : '$\\delta / \\omega_{mL}$',
        'x_tick_position'   : 'both-out',
        'x_ticks'           : [-0.02, 0.0, 0.02],
        'x_ticks_minor'     : [i * 0.004 - 0.02 for i in range(11)],
        'y_label'           : '$\\lambda / \\kappa$',
        'y_tick_labels'     : [0.0, 0.1, 0.2],
        'y_tick_position'   : 'both-out',
        'y_ticks'           : [0.0, 0.03, 0.06],
        'y_ticks_minor'     : [i * 0.006 for i in range(11)],
        'show_cbar'         : True,
        'cbar_title'        : '$\\langle S_{p} \\rangle$',
        'cbar_ticks'        : [0.0, 0.1, 0.2],
        'label_font_size'   : 24.0,
        'tick_font_size'    : 20.0,
        'title_font_size'   : 24.0,
        'width'             : 5.5,
        'annotations'       : [{
            'text'  : '(b)',
            'xy'    : [0.44, 0.91]
        }]
    }
}

# function to obtain quantum phase synchronization
def func(system_params):
    # get quantum correlation measures
    Measures = get_func_quantum_correlation_measures(
        SystemClass=Bi_00,
        params=params['solver'],
        steady_state=False
    )(system_params)
    # return average value
    return np.mean(Measures.transpose()[0])

# loop and plot
if __name__ == '__main__':
    looper = run_loopers_in_parallel(
        looper_name='XYLooper',
        func=func,
        params=params['looper'],
        params_system=params['system'],
        plot=True,
        params_plotter=params['plotter']
    )