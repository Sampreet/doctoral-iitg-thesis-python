# dependencies
import numpy as np
import os
import sys

# qom modules
from qom.solvers.deterministic import HLESolver
from qom.solvers.measure import QCMSolver
from qom.ui.plotters import MPLPlotter
from qom.utils.loopers import run_loopers_in_parallel, wrap_looper

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.OptoElectroMechanical import OEM_20

# all parameters
params = {
    'looper': {
        'show_progress' : True,
        'X'             : {
            'var'   : 'Omegas',
            'idx'   : 2,
            'min'   : 1.95,
            'max'   : 2.05,
            'dim'   : 1001
        }
    },
    'solver': {
        'show_progress' : False,
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
        'colors'                : ['k'] + [-1] * 2 + [0] * 2,
        'sizes'                 : [1] + [2] * 4,
        'styles'                : ['--'] + ['--', '-'] * 2,
        'x_label'               : '$\\Omega_{s} / \\omega_{b0}$',
        'x_ticks'               : [1.95 + i * 0.05 for i in range(3)],
        'x_ticks_minor'         : [1.95 + i * 0.0125 for i in range(9)],
        'v_label'               : '',
        'v_label_color'         : -1,
        'v_limits'              : [0.325, 0.675],
        'v_tick_color'          : 0,
        'v_tick_labels'         : [''] * 3,
        'v_tick_position'       : 'left-in',
        'v_ticks'               : [0.4, 0.5, 0.6],
        'v_ticks_minor'         : [0.325 + i * 0.025 for i in range(14)],
        'v_twin_label'          : '$E_{N_{\\mathrm{max}}}$',
        'v_twin_label_color'    : 0,
        'v_twin_limits'         : [-0.015, 0.195],
        'v_twin_tick_color'     : 0,
        'v_twin_tick_position'  : 'right-in',
        'v_twin_ticks'          : [0.03, 0.09, 0.15],
        'v_twin_ticks_minor'    : [i * 0.015 for i in range(14)],
        'label_font_size'       : 24,
        'tick_font_size'        : 20,
        'width'                 : 4.0,
        'height'                : 4.0,
        'annotations'           : [{
            'text'  : '(c)',
            'xy'    : [0.13, 0.84]
        }]
    }
}

# function to obtain squeezing and entanglement
def func(system_params):
    # initialize system
    system = OEM_20(
        params=system_params
    )

    # initialize solver
    hle_solver = HLESolver(
        system=system,
        params=params['solver']
    )
    # get modes and correlations
    Modes, Corrs = hle_solver.get_modes_corrs()
    # get quantum correlation measures
    Measures = QCMSolver(
        Modes=Modes,
        Corrs=Corrs,
        params=params['solver']
    ).get_measures()
    # extract maximum squeezing
    m_0 = np.min(Corrs[:, 2, 2])
    # extract maximum entanglement
    m_1 = np.max(Measures[:, 0])

    # return results as array
    return np.array([m_0, m_1], dtype=np.float_)

if __name__ == '__main__':
    # without voltage modulation
    params['looper']['file_path_prefix'] = 'data/v1.0_qom-v1.0.1/4.9c_A_vs=[50.0, 0.0, 0.0]'
    params['system']['A_vs'] = [50.0, 0.0, 0.0]
    looper_0 = run_loopers_in_parallel(
        looper_name='XLooper',
        func=func,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    Sq_0, En_0 = np.transpose(looper_0.results['V'])

    # with voltage modulation
    params['looper']['file_path_prefix'] = 'data/v1.0_qom-v1.0.1/4.9c_A_vs=[50.0, 50.0, 50.0]'
    params['system']['A_vs'] = [50.0, 50.0, 50.0]
    looper_1 = run_loopers_in_parallel(
        looper_name='XLooper',
        func=func,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    Sq_1, En_1 = np.transpose(looper_1.results['V'])

    # plotter
    X = looper_0.axes['X']['val']
    plotter = MPLPlotter(
        axes={},
        params=params['plotter']
    )
    # plot squeezing dynamics
    plotter.update(
        vs=[np.zeros(np.shape(X)) + 0.5, Sq_0, Sq_1],
        xs=X
    )
    # plot entanglement dynamics
    plotter.update_twin_axis(
        vs=[En_0, En_1],
        xs=X
    )
    # show
    plotter.show()