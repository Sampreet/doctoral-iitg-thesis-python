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
        'file_path_prefix'  : 'data/v1.0_qom-v1.0.1/3.7a',
        'grad'              : True,
        'X'                 : {
            'var'   : 'delta',
            'min'   : -0.002,
            'max'   : 0.002,
            'dim'   : 400001
        },
        'Y'                 : {
            'var'   : 'P_lc',
            'min'   : 0.0e-15,
            'max'   : 1.5e-15,
            'dim'   : 151
        },
        'Z'                 : {
            'var'   : 'L_p',
            'val'   : [1, 2, 3]
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
        'x_label'           : '$P_{l}$ (fW)',
        'x_tick_labels'     : [0.0, 0.5, 1.0, 1.5],
        'x_ticks'           : [i * 0.5e-15 for i in range(4)],
        'x_ticks_minor'     : [i * 0.125e-15 for i in range(13)],
        'y_name'            : '$L_{p}$',
        'v_label'           : '$\\tau_{g}$ (s)',
        'v_ticks'           : [i * 40 - 120 for i in range(4)],
        'v_ticks_minor'     : [i * 10 - 120 for i in range(13)],
        'show_legend'       : True,
        'legend_location'   : 'lower left',
        'label_font_size'   : 24.0,
        'legend_font_size'  : 24.0,
        'tick_font_size'    : 20.0,
        'width'             : 6.0,
        'height'            : 4.8,
        'annotations'       : [{
            'text'  : '(a)',
            'xy'    : (0.025, 0.875)
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
    # normalize by additive detuning
    _, Omegas, _, _ = system.get_effective_values(
        c=c
    )
    # return normalized value
    return phi / (Omegas[0] + Omegas[1]) * 2

if __name__ == '__main__':
    # looper
    looper = run_loopers_in_parallel(
        looper_name='XYZLooper',
        func=func_transmission_phase_norm,
        params=params['looper'],
        params_system=params['system']
    )
    xs = looper.axes['Y']['val']
    ys = looper.axes['Z']['val']
    vs = [[np.min(col) for col in row] for row in looper.results['V']]

    # plotter
    plotter = MPLPlotter(axes={
        'X': xs,
        'Y': ys
    }, params=params['plotter'])
    plotter.update(
        vs=vs,
        xs=xs
    )
    plotter.show()