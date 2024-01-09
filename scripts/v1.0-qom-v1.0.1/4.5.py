# dependencies
import numpy as np
import os
import sys

# qom modules
from qom.solvers.deterministic import HLESolver
from qom.ui.plotters import MPLPlotter
from qom.utils.loopers import run_loopers_in_parallel, wrap_looper

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.MiddleMembrane import MM_01

# all parameters
params = {
    'looper': {
        'show_progress' : True,
        'X'             : {
            'var'   : 'beta_pm_sum',
            'min'   : 75,
            'max'   : 225,
            'dim'   : 301
        },
        'Y'             : {
            'var'   : 'kappa_norm',
            'min'   : 1e-3,
            'max'   : 1e0,
            'dim'   : 31,
            'scale' : 'log'
        }
    },
    'solver': {
        'show_progress' : False,
        'cache'         : True,
        'ode_method'    : 'vode',
        'indices'       : [(2, 2)],
        't_min'         : 0.0,
        't_max'         : 10000.0,
        't_dim'         : 100001,
        't_index_min'   : 99371,
        't_index_max'   : 100001
    },
    'system': {
        'alphas'        : [2.0, 0.2, 0.2],
        'betas'         : [100.0, 25.0, 25.0],
        'Delta_norm'    : 1.0,
        'g_norm'        : -1e-4,
        'gamma_norm'    : 1e-6,
        'kappa_norm'    : 0.1,
        'ns'            : [0.0, 10.0],
        'Omega_norms'   : [2.0, 2.0],
        't_rwa'         : True
    },
    'plotter': {
        'type'              : 'lines',
        'colors'            : [-1, -1, 'k', 0, 0, 'k'],
        'sizes'             : [2.0, 2.0, 1.0] * 2,
        'styles'            : ['-', '-.', '--', '-', '-.', ':'],
        'x_label'           : '$\\kappa / \\omega_{m}$',
        'x_ticks'           : [10**(i - 2) for i in range(3)],
        'x_ticks_minor'     : sum([[10**(i - 2) * (j + 2) for i in range(2)] for j in range(8)], []),
        'x_scale'           : 'log',
        'v_label'           : '$- 10 \\mathrm{log}_{10} [ \\langle \\tilde{Q}^{2} \\rangle_{\\mathrm{min}} ]$',
        'v_label_color'     : -1,
        'v_tick_color'      : -1,
        'v_ticks'           : [i * 4 for i in range(6)],
        'v_ticks_minor'     : [i * 2 for i in range(11)],
        'v_twin_label_color': 0,
        'v_twin_tick_color' : 0,
        'v_twin_label'      : '$\\frac{G_{1}}{G_{0}} |_{\\mathrm{opt}}$',
        'v_twin_ticks'      : [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        'v_twin_ticks_minor': [i * 0.05 + 0.7 for i in range(11)],
        'show_legend'       : True,
        'legend_labels'     : [
            '$n_{b} = 10$',
            '$n_{b} = 1000$'
        ],
        'legend_location'   : 'upper right',
        'height'            : 4.0,
        'width'             : 8.0,
        'label_font_size'   : 24,
        'legend_font_size'  : 24,
        'tick_font_size'    : 20
    }
}

# function to calculate the ratio and variances
def func_rat_var(system_params):
    # update parameters
    val = system_params['beta_pm_sum']
    system_params['betas'][1] = val / 2.0
    system_params['betas'][2] = val / 2.0

    # initialize system
    system = MM_01(
        params=system_params
    )

    # get derived constants and controls
    _, _, c = system.get_ivc()
    
    # get squeezing ratio
    rat = system.get_params_ratio(
        c=c
    )

    # get mechanical position variances
    var = np.min(HLESolver(
        system=system,
        params=params['solver']
    ).get_corr_indices(), axis=0)[0]

    # update results
    return np.array([rat, var], dtype=np.float_)

if __name__ == '__main__':
    # low thermal phonons
    params['looper']['file_path_prefix'] = 'data/v1.0-qom-v1.0.1/4.5_n=10.0'
    params['system']['ns'][1] = 10.0
    looper = run_loopers_in_parallel(
        looper_name='XYLooper',
        func=func_rat_var,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    xs = looper.axes['Y']['val']
    _, vs_0 = np.min(looper.results['V'], axis=1).transpose()
    _, _idxs_0 = np.argmin(looper.results['V'], axis=1).transpose()
    vs_2 = np.transpose(looper.results['V'])[0, _idxs_0, 0]

    # high thermal phonons
    params['looper']['file_path_prefix'] = 'data/v1.0-qom-v1.0.1/4.5_n=1000.0'
    params['system']['ns'][1] = 1000.0
    looper = run_loopers_in_parallel(
        looper_name='XYLooper',
        func=func_rat_var,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    _, vs_1 = np.min(looper.results['V'], axis=1).transpose()
    _, _idxs_1 = np.argmin(looper.results['V'], axis=1).transpose()
    vs_3 = np.transpose(looper.results['V'])[0, _idxs_1, 0]

    # plotter
    plotter = MPLPlotter(
        axes={},
        params=params['plotter']
    )
    plotter.update(
        vs=- 10 * np.log10([vs_0, vs_1, [0.5] * len(vs_0)]),
        xs=xs
    )
    plotter.update_twin_axis(
        vs=[vs_2, vs_3, [1.0] * len(vs_0)],
        xs=xs
    )
    plotter.show(
        hold=True
    )