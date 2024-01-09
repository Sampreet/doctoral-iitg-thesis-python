# dependencies
import numpy as np
import os 
import sys

# qom modules
from qom.ui import init_log
from qom.solvers.deterministic import HLESolver
from qom.ui.plotters import MPLPlotter

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.EndMirror import EM_00

# all parameters
params = {
    'solver': {
        'show_progress' : True,
        'cache'         : True,
        'ode_method'    : 'vode',
        'indices'       : [1],
        't_min'         : 0.0,
        't_max'         : 1000.0,
        't_dim'         : 10001
    },
    'system': {
        'A_l_norm'      : 50.0,
        'Delta_0_norm'  : 1.0, 
        'g_0_norm'      : 0.005,
        'gamma_norm'    : 0.005,
        'kappa_norm'    : 0.15,
        'T_norm'        : 0.0
    },
    'plotter': {
        'type'              : 'lines',
        'colors'            : [0, -1],
        'sizes'             : [0.2, 2],
        'styles'            : ['-', '-'],
        'x_label'           : '$\\bar{q}$',
        'x_ticks'           : list(range(-800, 801, 400)),
        'x_ticks_minor'     : list(range(-800, 801, 100)),
        'v_label'           : '$\\bar{p}$',
        'v_ticks'           : list(range(-800, 801, 400)),
        'v_ticks_minor'     : list(range(-800, 801, 100)),
        'width'             : 5.0,
        'height'            : 4.7,
        'label_font_size'   : 24.0,
        'legend_font_size'  : 20.0,
        'tick_font_size'    : 20.0,
        'annotations'       : [{
            'text'  : '(b)',
            'xy'    : (0.29, 0.83)
        }]
    }
}
# initialize log
init_log()
# initialize system
system = EM_00(
    params=params['system']
)
# initialize solver
hle_solver = HLESolver(
    system=system,
    params=params['solver']
)
# get times and modes
T = hle_solver.T
Modes = hle_solver.get_mode_indices()
# extract position and momentum
q_fp = np.sqrt(2.0) * np.real(Modes.transpose()[0])
p_fp = np.sqrt(2.0) * np.imag(Modes.transpose()[0])
# initialize plotter
plotter = MPLPlotter(
    axes={},
    params=params['plotter']
)
# update and show
plotter.update(
    vs=[p_fp, p_fp[-628:]],
    xs=[q_fp, q_fp[-628:]]
)
plotter.add_scatter([0], [0], color=3, size=50, style='o')
plotter.show()