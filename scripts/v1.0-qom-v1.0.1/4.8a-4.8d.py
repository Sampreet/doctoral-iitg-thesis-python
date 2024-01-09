# dependencies
import numpy as np
import os
import sys

# qom modules
from qom.solvers.deterministic import HLESolver
from qom.solvers.measure import QCMSolver
from qom.ui import init_log
from qom.ui.plotters import MPLPlotter

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.OptoElectroMechanical import OEM_20

# all parameters
params = {
    'solver': {
        'show_progress' : True,
        'cache'         : True,
        'measure_codes' : ['entan_ln'],
        'indices'       : (0, 2),
        'ode_method'    : 'vode',
        't_min'         : 0.0,
        't_max'         : 1000.0,
        't_dim'         : 10001,
        't_index_min'   : 9371,
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
        'type'                  : 'lines',
        'colors'                : ['k', -1, 0],
        'sizes'                 : [1, 2, 2],
        'x_label'               : '$\\omega_{b0} t$',
        'x_ticks'               : [990, 995, 1000],
        'x_ticks_minor'         : [990 + i for i in range(11)],
        'v_label_color'         : -1,
        'v_limits'              : [0.35, 0.9],
        'v_tick_color'          : -1,
        'v_tick_position'       : 'left-in',
        'v_ticks'               : [0.4, 0.6, 0.8],
        'v_ticks_minor'         : [0.35 + i * 0.05 for i in range(13)],
        'v_twin_label_color'    : 0,
        'v_twin_limits'         : [0.015, 0.18],
        'v_twin_tick_color'     : 0,
        'v_twin_tick_position'  : 'right-in',
        'v_twin_ticks'          : [0.03, 0.09, 0.15],
        'v_twin_ticks_minor'    : [0.015 + i * 0.015 for i in range(13)],
        'label_font_size'       : 24.0,
        'tick_font_size'        : 20.0,
        'width'                 : 3.6,
        'height'                : 3.2
    }
}

# initialize logger
init_log()

# variable plotter parameters
annotation_texts = ['(a)', '(b)', '(c)', '(d)']
annotation_xs = [0.34, 0.13, 0.34, 0.13]
styles = [':', '--', '-.', '-']

for j in range(4):
    # update sytem params
    params['system']['A_vs'] = [50.0, 0.0, 0.0] if int(j / 2) == 0 else [50.0, 50.0, 50.0]
    params['system']['theta'] = 0.0 if j % 2 == 0 else 0.5

    # initialize system
    system = OEM_20(
        params=params['system']
    )

    # initialize solver
    hle_solver = HLESolver(
        system=system,
        params=params['solver']
    )
    # get times, modes and correlations
    T = hle_solver.get_times()
    Modes = hle_solver.get_modes()
    Corrs = hle_solver.get_corrs()
    # get quantum correlation measures
    Measures = QCMSolver(
        Modes=Modes,
        Corrs=Corrs,
        params=params['solver']
    ).get_measures()
    # extract correlation
    M_0 = Corrs[:, 2, 2]
    # extract entanglement
    M_1 = Measures[:, 0]
    # output maximum squeezing and entanglement
    print(np.min(M_0), np.max(M_1))

    # update plotter params
    params['plotter']['styles'] = ['--', styles[j], styles[j]]
    params['plotter']['v_tick_labels'] = [0.4, 0.6, 0.8] if j % 2 == 0 else [''] * 3
    params['plotter']['v_label'] = '$\\langle Q_{b}^{2} \\rangle$' if j % 2 == 0 else ''
    params['plotter']['v_twin_tick_labels'] = [0.03, 0.09, 0.15] if j % 2 == 1 else [''] * 3
    params['plotter']['v_twin_label'] = '$E_{N}$' if j % 2 == 1 else ''
    params['plotter']['annotations'] = [{
        'text': annotation_texts[j],
        'xy': [annotation_xs[j], 0.83]
    }]
    params['plotter']['vertical_spans'] = list()
    # calculate areas near 25% of the minimum variance
    idxs = np.squeeze(np.argwhere(M_0 < np.min(M_0) + 0.25 * (np.max(M_0) - np.min(M_0))))
    jumps = idxs[:-1] != idxs[1:] - 1
    stops = T[idxs][:-1][jumps]
    starts = T[idxs][1:][jumps]
    for i in range(1, len(starts)):
        params['plotter']['vertical_spans'].append({
            'limits': (starts[i], (stops[i + 1] if i < len(stops) - 1 else T[idxs][-1])),
            'color': 8,
            'alpha': 0.25
        })
    # calculate areas near 25% of the maximum entanglement
    idxs = np.squeeze(np.argwhere(M_1 > np.max(M_1) - 0.25 * (np.max(M_1) - np.min(M_1))))
    jumps = idxs[:-1] != idxs[1:] - 1
    stops = T[idxs][:-1][jumps]
    starts = T[idxs][1:][jumps]
    for i in range(1, len(starts)):
        params['plotter']['vertical_spans'].append({
            'limits': (starts[i], (stops[i + 1] if i < len(stops) - 1 else T[idxs][-1])),
            'color': 2,
            'alpha': 0.25
        })

    # plotter
    plotter = MPLPlotter(
        axes={},
        params=params['plotter']
    )
    # plot correlation dynamics
    plotter.update(
        vs=[np.zeros(np.shape(T)) + 0.5, M_0],
        xs=T
    )
    # plot entanglement dynamics
    plotter.update_twin_axis(
        vs=M_1,
        xs=T
    )
    # show
    plotter.show()
    # close
    plotter.close()