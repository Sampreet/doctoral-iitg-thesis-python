# dependencies
import numpy as np
import os
import sys

# qom modules
from qom.solvers.deterministic import HLESolver
from qom.solvers.measure import get_Wigner_distributions_single_mode
from qom.ui import init_log
from qom.ui.plotters import MPLPlotter

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.EndMirror import EM_00

# frequently used variables
_max = 3
_dim = 101

# all parameters
params = {
    'solver': {
        'show_progress' : True,
        'cache'         : True,
        'ode_method'    : 'vode',
        'indices'       : [1],
        'wigner_xs'     : np.linspace(-_max, _max, _dim),
        'wigner_ys'     : np.linspace(-_max, _max, _dim),
        't_min'         : 0.0,
        't_max'         : 1000.0,
        't_dim'         : 100001,
        't_index_min'   : 99810,
        't_index_max'   : 99810,
    },
    'system': {
        'A_l_norm'      : 10.0,
        'Delta_0_norm'  : 1.0,
        'g_0_norm'      : 0.0001,
        'gamma_norm'    : 0.0005,
        'kappa_norm'    : 0.15,
        'T_norm'        : 0.0
    },
    'plotter': {
        'type'              : 'surface_cz',
        'palette'           : 'Reds',
        'bins'              : 11,
        'x_label'           : '$Q$',
        'x_tick_position'   : 'both-out',
        'x_limits'          : [-_max - _max / 10, _max + _max / 10],
        'x_ticks'           : [-_max, 0, _max],
        'x_ticks_minor'     : [i - _max for i in range(7)],
        'x_tick_pad'        : 0,
        'y_limits'          : [-_max - _max / 10, _max + _max / 10],
        'y_label'           : '$P$',
        'y_tick_position'   : 'both-out',
        'y_ticks'           : [-_max, 0, _max],
        'y_ticks_minor'     : [i - _max for i in range(7)],
        'y_tick_pad'        : 0,
        'v_ticks_minor'     : [i * 0.05 for i in range(8)],
        'v_label'           : '$W$',
        'v_label_pad'       : 12,
        'v_tick_pad'        : 8,
        'v_ticks'           : [0.0, 0.2, 0.4],
        'label_font_size'   : 21.0,
        'tick_font_size'    : 18.0,
        'title_font_size'   : 21.0,
        'width'             : 4.8,
        'height'            : 4.5,
        'annotations'       : [{
            'text'  : '(b)',
            'xy'    : (0.175, 0.82)
        }]
    }
}

# initialize logger
init_log()

# initialize system
system = EM_00(
    params=params['system']
)

# get correlations and times
hle_solver = HLESolver(
    system=system,
    params=params['solver']
)
T = hle_solver.get_times()
Corrs = hle_solver.get_corrs()

# get Wigner distributions
Wigners = get_Wigner_distributions_single_mode(
    Corrs=Corrs,
    params=params['solver']
)
# plotter
for i in range(0, len(T), 10000):
    print(Corrs[i, 2, 2], Corrs[i, 3, 3], - 10 * np.log10(Corrs[i, 2, 2]))
    plotter = MPLPlotter(
        axes={
            'X': np.linspace(-_max, _max, _dim),
            'Y': np.linspace(-_max, _max, _dim)
        },
        params=params['plotter']
    )
    plotter.update(
        vs=Wigners[i, 0]
    )
    plotter.show()