# dependencies
import numpy as np
import os
import sys

# qom modules
from qom.utils.loopers import wrap_looper

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.EndMirror import EM_00

# function
def func(system_params):
    # initialize system
    system = EM_00(
        params=system_params
    )

    # frequently used parameters
    Delta_0_norm = system.params['Delta_0_norm']
    g_0_norm = system.params['g_0_norm']
    gamma_norm = system.params['gamma_norm']
    kappa_norm = system.params['kappa_norm']
    delta_norm = system_params['delta_norm']
    Kappa_p_norm = kappa_norm / 2.0 + 1.0j * Delta_0_norm - 1j * delta_norm
    Kappa_m_norm = kappa_norm / 2.0 - 1.0j * Delta_0_norm - 1j * delta_norm
    Gamma_2_norm = (1.0 - Delta_0_norm) * (1.0 - delta_norm) + gamma_norm**2 / 4 + 1j * Delta_0_norm * gamma_norm

    # mean optical occupancies
    N_os = system.get_mean_optical_occupancies()

    # anti-Stokes coefficient
    num = Kappa_p_norm * Gamma_2_norm + 2j * g_0_norm**2 * 1.0 * N_os[0]
    den = Kappa_p_norm * Kappa_m_norm * Gamma_2_norm + 4 * g_0_norm**2 * Delta_0_norm * 1.0 * N_os[0]

    # transmission
    return np.abs(1 - kappa_norm * num / den / 2)

# looper
A_ls = np.flip(list(range(0, 101, 25)))
deltas = np.linspace(0.5, 1.5, 10001)
looper = wrap_looper(
    looper_name = 'XYLooper',
    func=func,
    params={
        'show_progress' : True,
        'X'             : {
            'var'   : 'delta_norm',
            'val'   : deltas
        },
        'Y'             : {
            'var'   : 'A_l_norm',
            'val'   : A_ls
        }
    },
    params_system={
        'A_l_norm'      : 5.0,
        'delta_norm'    : 1.0,
        'Delta_0_norm'  : -1.0, 
        'g_0_norm'      : 0.0005,
        'gamma_norm'    : 0.005,
        'kappa_norm'    : 0.15,
        'T_norm'        : 0.0
    },
    plot=True,
    params_plotter={
        'type'              : 'lines_3d',
        'palette'           : 'tab10',
        'bins'              : 10,
        'colors'            : [0, 2, 3, 5, 7],
        'sizes'             : [2] * 5,
        'styles'            : ['-', ':', '-.', '--', '-'],
        'x_label'           : '$\\delta / \\omega_{m}$',
        'x_label_pad'       : 2,
        'x_tick_pad'        : -2,
        'x_ticks'           : [deltas[0], 1, deltas[-1]],
        'x_ticks_minor'     : [i * 0.25 + deltas[0] for i in range(5)],
        'y_label'           : '$A_{l} / \\omega_{m}$',
        'y_label_pad'       : 16,
        'y_tick_pad'        : 0,
        'y_ticks_minor'     : A_ls,
        'y_ticks'           : [0, 50, 100],
        'y_ticks_minor'     : [i * 10 for i in range(11)],
        'v_label'           : '$T$',
        'v_label_pad'       : -12,
        'v_limits'          : [-0.1, 1.1],
        'v_tick_labels'     : [0.0, '', 1.0],
        'v_tick_pad'        : 2,
        'v_ticks'           : [0.0, 0.5, 1.0],
        'v_ticks_minor'     : [i * 0.25 for i in range(5)],
        'height'            : 4.5,
        'width'             : 4.2,
        'view_aspect'       : [1.0, 1.5, 0.75],
        'view_elevation'    : 32,
        'view_rotation'     : -50,
        'label_font_size'   : 24.0,
        'tick_font_size'    : 20.0,
        'annotations'       : [{
            'text'  : '(b)',
            'xy'    : (0.1, 0.75)
        }]
    }
)

