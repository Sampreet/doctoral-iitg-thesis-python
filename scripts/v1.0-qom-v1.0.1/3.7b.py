# dependencies
import numpy as np
import os 
import sys

# qom modules
from qom.ui.plotters import MPLPlotter
from qom.utils.loopers import run_loopers_in_parallel, wrap_looper

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.BoseEinsteinCondensate import BEC_10

# all parameters
params = {
    'looper': {
        'show_progress'     : True,
        'file_path_prefix'  : 'data/v1.0-qom-v1.0.1/3.7b',
        'grad'              : True,
        'X'                 : {
            'var'   : 'delta',
            'min'   : -0.002,
            'max'   : 0.002,
            'dim'   : 400001
        },
        'Y'                 : {
            'var'   : 'L_p',
            'min'   : 0,
            'max'   : 10,
            'dim'   : 101
        },
        'Z'                 : {
            'var'   : 'P_lc',
            'val'   : [0.5e-15, 1e-15, 1.5e-15, 2e-15]
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
        't_approx'      : 'res',
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
        'palette'           : 'tab10',
        'bins'              : 10,
        'colors'            : [0, 2, 3, 5, 7],
        'sizes'             : [2] * 5,
        'styles'            : ['-', ':', '-.', '--', '-'],
        'x_label'           : '$L_{p}$',
        'x_ticks'           : [i * 2 for i in range(6)],
        'x_ticks_minor'     : [i * 0.5 for i in range(21)],
        'y_name'            : '$P_{l}$',
        'y_unit'            : 'fW',
        'v_label'           : '$\\tau_{g}$ (s)',
        'v_ticks'           : [-i * 50 for i in range(4)],
        'v_ticks_minor'     : [-i * 10 for i in range(16)],
        'show_legend'       : True,
        'legend_location'   : 'lower left',
        'legend_font_size'  : 24.0,
        'label_font_size'   : 24.0,
        'tick_font_size'    : 20.0,
        'height'            : 4.8,
        'width'             : 6.0,
        'annotations'       : [{
            'text'  : '(b)',
            'xy'    : (0.025, 0.875)
        }],
    }
}

# function to calculate normalized transmission phase
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
    # normalize by additive detuning
    _, Omegas, _, _ = system.get_effective_values(
        c=c
    )
    # return normalized value
    return phi / (Omegas[0] + Omegas[1]) * 2

if __name__ == '__main__':
    # looper
    looper = wrap_looper(
        looper_name='XYZLooper',
        func=func_transmission_phase_norm,
        params=params['looper'],
        params_system=params['system']
    )
    xs = looper.axes['Y']['val']
    vs = [[np.min(col) for col in row] for row in looper.results['V']]

    # plotter
    plotter = MPLPlotter(axes={
        'X': xs,
        'Y': [0.5, 1.0, 1.5, 2.0]
    }, params=params['plotter'])
    plotter.update(
        vs=vs,
        xs=xs
    )
    plotter.show()