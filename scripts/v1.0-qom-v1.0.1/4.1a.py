# dependencies
import numpy as np
import os
import sys

# qom modules
from qom.solvers.deterministic import HLESolver
from qom.ui.plotters import MPLPlotter
from qom.ui import init_log

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.EndMirror import EM_00, EM_01

# all parameters
params = {
    'solver': {
        'show_progress' : True,
        'cache'         : True,
        'ode_method'    : 'vode',
        't_min'         : 0.0,
        't_max'         : 300.0,
        't_dim'         : 3001
    },
    'system': {
        'A_l_norms'     : [25.0, 2.5, 2.5],
        'Delta_0_norm'  : -1.0, 
        'g_0_norm'      : 0.005,
        'gamma_norm'    : 0.005,
        'kappa_norm'    : 0.15,
        'T_norm'        : 0.0
    },
    'plotter': {
        'type'              : 'lines',
        'colors'            : [2, 0, -3, -1],
        'sizes'             : [2.0, 2.0] * 2,
        'styles'            : ['-', ':'] * 2,
        'x_label'           : '$\\omega_{m} t$',
        'x_ticks'           : [i * 50 for i in range(7)],
        'x_ticks_minor'     : [i * 10 for i in range(31)],
        'v_label_color'     : 0,
        'v_tick_color'      : 0,
        'v_label'           : '$| \\alpha | / 10^{3}$',
        'v_tick_labels'     : ['{:0.1f}'.format(i * 0.4) for i in range(4)],
        'v_ticks'           : list(range(0, 1201, 400)),
        'v_ticks_minor'     : list(range(0, 2001, 100)),
        'v_twin_label_color': -1,
        'v_twin_tick_color' : -1,
        'v_twin_label'      : '$\\beta^{*} + \\beta$',
        'v_twin_ticks'      : list(range(6, 10, 1)),
        'v_twin_ticks_minor': [i * 0.25 + 6 for i in range(17)],
        'show_legend'       : True,
        'legend_labels'     : [
            'with modulation',
            'without modulation',
        ],
        'legend_location'   : 'upper center',
        'width'             : 9.6,
        'height'            : 4.0,
        'label_font_size'   : 24.0,
        'legend_font_size'  : 20.0,
        'tick_font_size'    : 20.0,
        'annotations'       : [{
            'text'  : '(a)',
            'xy'    : (0.86, 0.83)
        }]
    }
}

# init log
init_log()

# initialize system
system = EM_01(
    params=params['system']
)

# initialize solver
solver = HLESolver(
    system=system,
    params=params['solver']
)
# get times
T = solver.get_times()
Modes = solver.get_modes()
# get modes
vs_0 = [np.real(np.conjugate(alpha) * alpha) for alpha in Modes[:, 0]]
vs_1 = [2 * np.real(beta) for beta in Modes[:, 1]]

# initialize system
system = EM_00(
    params=params['system']
)

# initialize solver
solver = HLESolver(
    system=system,
    params=params['solver']
)
# get times
T = solver.get_times()
Modes = solver.get_modes()
# get modes
vs_2 = [np.real(np.conjugate(alpha) * alpha) for alpha in Modes[:, 0]]
vs_3 = [2 * np.real(beta) for beta in Modes[:, 1]]

# plotter
plotter = MPLPlotter(
    axes={},
    params=params['plotter']
)
plotter.update(
    vs=[vs_0, vs_2],
    xs=T
)
plotter.update_twin_axis(
    vs=[vs_1, vs_3],
    xs=T
)
plotter.show()