# dependencies
import numpy as np
import os 
import sys

# qom modules
from qom.utils.loopers import wrap_looper

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.BoseEinsteinCondensate import BEC_10

# all parameters
params = {
    'looper': {
        'show_progress'     : True,
        'file_path_prefix'  : 'data/v1.0-qom-v1.0.1/3.4a',
        'X'                 : {
            'var'   : 'delta',
            'min'   : -0.5,
            'max'   : 0.5,
            'dim'   : 10001
        },
        'Y'                 : {
            'var'   : 'P_lc',
            'val'   : [1.5e-15, 1.0e-15, 0.5e-15, 0.05e-15]
        }
    },
    'system': {
        'Delta_tilde'   : -1.0,
        'delta'         : 0.0,
        'G'             : 2 * np.pi * 1e3,
        'g_tilde_norm'  : 0.0,
        'gamma_m'       : 2 * np.pi * 0.8,
        'gamma_o'       : 2 * np.pi * 1e3,
        'k'             : 1,
        'L_p'           : 0,
        'l'             : 20,
        'lambda_lc'     : 589e-9,
        'm'             : 23,
        'mu'            : 0.5,
        'N'             : 1e4,
        'P_lc'          : 1e-15,
        'P_lp_norm'     : 0.01,
        'R'             : 10e-6,
        't_approx'      : 'none',
        't_Delta_norm'  : 'Omega_m',
        't_Delta_offset': 'zero',
        't_delta_norm'  : 'Omega_m',
        't_delta_offset': 'Omega_m',
        't_line'        : 's',
        't_oss_method'  : 'cubic',
        't_P_lc_norm'   : 'none'
    },
    'plotter': {
        'type'              : 'lines_3d',
        'palette'           : 'tab10',
        'bins'              : 10,
        'colors'            : [2, 3, 5, 7],
        'sizes'             : [2] * 4,
        'styles'            : [':', '-.', '--', '-'],
        'x_label_pad'       : 32,
        'x_label'           : '$\\delta / \\Omega_{m}$',
        'x_label_pad'       : 8,
        'x_tick_pad'        : -2,
        'x_limits'          : [-0.55, 0.55],
        'x_tick_labels'     : [0.5, 1.0, 1.5],
        'x_ticks'           : [-0.5, 0.0, 0.5],
        'x_ticks_minor'     : [i * 0.25 - 0.5 for i in range(5)],
        'y_label'           : '$P_{l}$ (fW)',
        'y_limits'          : [-0.1e-15, 1.7e-15],
        'y_tick_labels'     : [0.05, 0.5, 1.0, 1.5],
        'y_label_pad'       : 24,
        'y_tick_pad'        : 4,
        'y_ticks'           : [0.05e-15, 0.5e-15, 1.0e-15, 1.5e-15],
        'y_ticks_minor'     : [0.0, 0.25e-15, 0.75e-15, 1.25e-15],
        'v_label_pad'       : 7,
        'v_label'           : '$T$',
        'v_tick_pad'        : 6,
        'v_ticks'           : [0.0, 0.5, 1.0],
        'v_ticks_minor'     : [i * 0.1 for i in range(11)],
        'view_aspect'       : [1.0, 1.5, 0.75],
        'view_elevation'    : 32,
        'view_rotation'     : -50,
        'annotations'       : [{
            'text'  : '(a)',
            'xy'    : (0.12, 0.72)
        }],
        'label_font_size'   : 24.0,
        'tick_font_size'    : 20.0,
        'height'            : 6.0,
        'width'             : 6.5
    }
}

# function to obtain the transmission
def func(system_params):
    # intialize system
    system = BEC_10(
        params=system_params
    )
    _, _, c = system.get_ivc()
    # return transmission
    return system.get_transmission(
        c=c
    )

# looper
looper = wrap_looper(
    looper_name='XYLooper',
    func=func,
    params=params['looper'],
    params_system=params['system'],
    plot=True,
    params_plotter=params['plotter']
)