# dependencies
import numpy as np
import os
import sys

# qom modules
from qom.solvers.deterministic import HLESolver
from qom.solvers.measure import QCMSolver
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
        't_max'         : 500.0,
        't_dim'         : 5001,
        'measure_codes' : ['entan_ln'],
        'indices'       : [0, 1]
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
        'colors'            : [-3, -1, 2, 0],
        'sizes'             : [2.0, 2.0] * 2,
        'styles'            : ['-', ':'] * 2,
        'x_label'           : '$\\omega_{m} t$',
        'x_ticks'           : [i * 10 + 300 for i in range(4)],
        'x_ticks_minor'     : [i * 2 + 300 for i in range(16)],
        'v_label_color'     : -1,
        'v_tick_color'      : -1,
        'v_label'           : '$\\langle Q^{2} \\rangle$',
        'v_tick_labels'     : ['{:0.1f}'.format(i * 0.1 + 0.4) for i in range(4)],
        'v_ticks'           : [i * 0.1 + 0.4 for i in range(4)],
        'v_ticks_minor'     : [i * 0.025 + 0.4 for i in range(13)],
        'show_legend'       : True,
        'legend_labels'     : [
            'with modulation',
            'without modulation',
        ],
        'legend_location'   : 'upper left',
        'label_font_size'   : 24.0,
        'legend_font_size'  : 20.0,
        'tick_font_size'    : 20.0,
        'width'             : 4.8,
        'height'            : 4.0,
        'annotations'       : [{
            'text'  : '(b)',
            'xy'    : (0.81, 0.83)
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
hle_solver = HLESolver(
    system=system,
    params=params['solver']
)
# get times
T = hle_solver.get_times()
Modes = hle_solver.get_modes()
Corrs = hle_solver.get_corrs()
# variances
vs_0 = Corrs[:, 2, 2]
# entanglement
vs_1 = QCMSolver(
    Modes=Modes,
    Corrs=Corrs,
    params=params['solver']
).get_measures()[:, 0]

# initialize system
system = EM_00(
    params=params['system']
)

# initialize solver
hle_solver = HLESolver(
    system=system,
    params=params['solver']
)
# get times
T = hle_solver.get_times()
Modes = hle_solver.get_modes()
Corrs = hle_solver.get_corrs()
# variances
vs_2 = Corrs[:, 2, 2]
# entanglement
vs_3 = QCMSolver(
    Modes=Modes,
    Corrs=Corrs,
    params=params['solver']
).get_measures()[:, 0]

# plotter
plotter = MPLPlotter(
    axes={},
    params=params['plotter']
)
plotter.update(
    vs=[vs_0, vs_2],
    xs=T
)
plotter.show()