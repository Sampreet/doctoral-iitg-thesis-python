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
            'var'   : 'ns',
            'idx'   : 1,
            'min'   : 1e-3,
            'max'   : 1e5,
            'dim'   : 81,
            'scale' : 'log'
        }
    },
    'solver': {
        'show_progress' : False,
        'cache'         : True,
        'ode_method'    : 'vode',
        'measure_codes' : ['entan_ln'],
        'indices'       : (0, 1),
        't_min'         : 0.0,
        't_max'         : 1000.0,
        't_dim'         : 10001,
        't_index_min'   : 9371,
        't_index_max'   : 10001
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
        'colors'            : [-1, -1, 'k', 0, 0],
        'sizes'             : [2.0] * 2 + [1.0] + [2.0] * 2,
        'styles'            : ['-', '-.', '--', '-', '-.'],
        'x_label'           : '$n_{b}$',
        'x_tick_labels'     : ['$10^{' + str(i - 3) + '}$' for i in range(9)],
        'x_ticks'           : [10**(i - 3) for i in range(9)],
        'x_ticks_minor'     : sum([[10**(i - 3) * (j + 2) for i in range(8)] for j in range(7)], []),
        'x_scale'           : 'log',
        'v_label'           : '$- 10 \\mathrm{log}_{10} [ \\langle \\tilde{Q}^{2} \\rangle_{\\mathrm{min}} ]$',
        'v_label_color'     : -1,
        'v_tick_color'      : -1,
        'v_ticks'           : [i * 4 for i in range(6)],
        'v_ticks_minor'     : [i * 2 for i in range(11)],
        'v_twin_label_color': 0,
        'v_twin_tick_color' : 0,
        'v_twin_label'      : '$E_{N_{\\mathrm{max}}}$',
        'v_twin_tick_labels': ['{:0.1f}'.format(i * 0.1) for i in range(6)],
        'v_twin_ticks'      : [i * 0.1 for i in range(6)],
        'v_twin_ticks_minor': [i * 0.05 for i in range(11)],
        'show_legend'       : True,
        'legend_labels'     : [
            '$\\kappa = 0.1 \\omega_{m}$',
            '$\\kappa = 1.0 \\omega_{m}$'
        ],
        'legend_location'   : 'upper right',
        'label_font_size'   : 24,
        'legend_font_size'  : 24,
        'tick_font_size'    : 20,
        'height'            : 4.0,
        'width'             : 8.0
    }
}

# function to calculate the ratio and entanglement
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

    # get mechanical position variance
    var = np.mean(HLESolver(
        system=system,
        params=params['solver']
    ).get_corrs()[:, 2, 2])

    return np.array([rat, var])

# function to calculate the ratio and entanglement
def func_rat_entan_ln(system_params):
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

    # get modes, correlations and times
    Modes, Corrs = HLESolver(
        system=system,
        params=params['solver']
    ).get_modes_corrs()
    # get entanglement
    eln = np.mean(QCMSolver(
        Modes=Modes,
        Corrs=Corrs,
        params=params['solver']
    ).get_measures(), axis=0)[0]

    return np.array([rat, eln])

if __name__ == '__main__':
    # low kappa
    params['looper']['file_path_prefix'] = 'data/v1.0_qom-v1.0.1/4.6_var_kappa=0.1'
    params['system']['kappa_norm'] = 0.1
    looper = run_loopers_in_parallel(
        looper_name='XYLooper',
        func=func_rat_var,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    xs = looper.axes['Y']['val']
    vars_0 = np.min(looper.results['V'], axis=1).transpose()[1]

    # high kappa
    params['looper']['file_path_prefix'] = 'data/v1.0_qom-v1.0.1/4.6_var_kappa=1.0'
    params['system']['kappa_norm'] = 1.0
    looper = run_loopers_in_parallel(
        looper_name='XYLooper',
        func=func_rat_var,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    vars_1 = np.min(looper.results['V'], axis=1).transpose()[1]
    
    # low kappa
    params['looper']['file_path_prefix'] = 'data/v1.0_qom-v1.0.1/4.6_entan_kappa=0.1'
    params['system']['kappa_norm'] = 0.1
    looper = run_loopers_in_parallel(
        looper_name='XYLooper',
        func=func_rat_entan_ln,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    xs = looper.axes['Y']['val']
    elns_0 = np.max(looper.results['V'], axis=1).transpose()[1]

    # high kappa
    params['looper']['file_path_prefix'] = 'data/v1.0_qom-v1.0.1/4.6_entan_kappa=1.0'
    params['system']['kappa_norm'] = 1.0
    looper = run_loopers_in_parallel(
        looper_name='XYLooper',
        func=func_rat_entan_ln,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    elns_1 = np.max(looper.results['V'], axis=1).transpose()[1]

    # plotter
    plotter = MPLPlotter(
        axes={},
        params=params['plotter']
    )
    plotter.update(
        vs=- 10 * np.log10([vars_0, vars_1, [0.5] * len(vars_0)]),
        xs=xs
    )
    plotter.update_twin_axis(
        vs=[elns_0, elns_1],
        xs=xs
    )
    plotter.show()