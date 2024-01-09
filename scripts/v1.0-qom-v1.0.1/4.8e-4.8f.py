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
from systems.OptoElectroMechanical import OEM_20

# parameters
params = {
    'solver': {
        'show_progress' : True,
        'cache'         : True,
        'ode_method'    : 'vode',
        'indices'       : [1],
        'wigner_xs'     : np.linspace(-2.0, 2.0, 401),
        'wigner_ys'     : np.linspace(-2.0, 2.0, 401),
        't_min'         : 0.0,
        't_max'         : 1000.0,
        't_dim'         : 10001,
        't_index_min'   : 9900,
        't_index_max'   : 10001
    },
    'system': {
        'A_ls'      : [100.0, 10.0, 10.0],
        'A_vs'      : [50.0, 50.0, 50.0],
        'Delta_0'   : 1.0,
        'gammas'    : [0.1, 1e-6, 1e-2],
        'gs'        : [1e-3, 2e-4],
        'n_ths'     : [0.0, 0.0],
        'Omegas'    : [2.0, 2.0, 2.0],
        'omega_c0'  : 1.1,
        'theta'     : 0.5,
        't_mod'     : 'cos',
        't_pos'     : 'top'
    },
    'plotter': {
        'type'              : 'contourf',
        'title'             : '$\\omega_{b0} t = 991.4$',
        'x_label'           : '$Q_{b}$',
        'x_label_pad'       : -16,
        'x_tick_labels'     : [-2, '', 2],
        'x_tick_position'   : 'both-out',
        'x_ticks'           : [-2, 0, 2],
        'x_ticks_minor'     : [-2, -1, 0, 1, 2],
        'y_label'           : '$P_{b}$',
        'y_label_pad'       : -16,
        'y_tick_labels'     : [-2, '', 2],
        'y_tick_position'   : 'both-out',
        'y_ticks'           : [-2, 0, 2],
        'y_ticks_minor'     : [-2, -1, 0, 1, 2],
        'show_cbar'         : True,
        'cbar_title'        : '$W$',
        'cbar_ticks'        : [0.0, 0.1, 0.2, 0.3],
        'label_font_size'   : 24.0,
        'tick_font_size'    : 20.0,
        'title_font_size'   : 24.0,
        'width'             : 4.0,
        'height'            : 3.6
    }
}

# initialize logger
init_log()

# initialize system
system = OEM_20(
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

# plot squeezed Wigners
for i in [14, 91]:
    # update parameters and plot
    params['plotter']['title'] = '$\\omega_{b0} t = ' + str(T[i]) +'$'
    plotter = MPLPlotter(
        axes={
            'X': params['solver']['wigner_xs'],
            'Y': params['solver']['wigner_ys']
        },
        params=params['plotter']
    )
    plotter.update(
        vs=Wigners[i, 0]
    )
    plotter.show()
    plotter.close()