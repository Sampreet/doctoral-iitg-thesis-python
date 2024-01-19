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
        'file_path_prefix'  : 'data/v1.0_qom-v1.0.1/3.6a',
        'X'                 : {
            'var'   : 'delta',
            'min'   : -0.2,
            'max'   : 0.2,
            'dim'   : 10001
        },
        'Y'                 : {
            'var'   : 'L_p',
            'val'   : [0, 1]
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
        'L_p'           : 1,
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
        't_oss_method'  : 'basic',
        't_P_lc_norm'   : 'none'
    },
    'plotter': {
        'type'              : 'lines',
        'colors'            : [0, -1],
        'sizes'             : [2] * 2,
        'x_label'           : '$\\delta / \\Omega_{m}$',
        'x_tick_labels'     : [0.8, 0.9, 1.0, 1.1, 1.2],
        'x_ticks'           : [i * 0.1 - 0.2 for i in range(5)],
        'x_ticks_minor'     : [i * 0.02 - 0.2 for i in range(21)],
        'y_name'            : '$L_{p}$',
        'v_label'           : '$\\phi$',
        'v_limits'          : [-2.4, 2.4],
        'v_ticks'           : [i - 2 for i in range(5)],
        'v_ticks_minor'     : [i * 0.2 - 2.4 for i in range(25)],
        'show_legend'       : True,
        'legend_location'   : 'lower left',
        'label_font_size'   : 24.0,
        'legend_font_size'  : 24.0,
        'tick_font_size'    : 20.0,
        'width'             : 4.8,
        'height'            : 4.2,
        'annotations'       : [{
            'text'  : '(a)',
            'xy'    : (0.225, 0.85)
        }],
    }
}

# function to obtain the normalized transmission phase
def func_transmission_phase_norm(system_params):
    # initialize system
    system = BEC_10(
        params=system_params
    )
    # extract parameters
    _, _, c = system.get_ivc()
    # get transmission phase
    phi = system.get_transmission_phase(
        c=c
    )
    # return normalized value
    return phi

# looper
looper = wrap_looper(
    looper_name='XYLooper',
    func=func_transmission_phase_norm,
    params=params['looper'],
    params_system=params['system'],
    plot=True,
    params_plotter=params['plotter']
)