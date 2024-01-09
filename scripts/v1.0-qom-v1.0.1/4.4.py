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
        }
    },
    'solver': {
        'show_progress' : False,
        'cache'         : True,
        'ode_method'    : 'vode',
        'indices'       : [(2, 2), (3, 3)],
        't_min'         : 0.0,
        't_max'         : 1000.0,
        't_dim'         : 10001,
        't_index_min'   : 9371,
        't_index_max'   : 10000
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
        'sizes'             : [2.0, 2.0, 1.0] * 2 ,
        'styles'            : ['-', '-.', '--', '-', '-.', ':'],
        'x_label'           : '$G_{1} / G_{0}$',
        'x_ticks'           : [i * 0.1 + 0.5 for i in range(6)],
        'x_ticks_minor'     : [i * 0.025 + 0.5 for i in range(21)],
        'v_label'           : '$- 10 \\mathrm{log}_{10} [ \\langle \\tilde{Q}^{2} \\rangle ]$',
        'v_label_color'     : -1,
        'v_tick_color'      : -1,
        'v_ticks'           : [i * 4 for i in range(6)],
        'v_ticks_minor'     : [i * 2 for i in range(11)],
        'v_twin_label'      : '$\\langle \\tilde{\\beta}^{\\dagger} \\tilde{\\beta} \\rangle$',
        'v_twin_scale'      : 'log',
        'v_twin_label_color': 0,
        'v_twin_tick_color' : 0,
        'v_twin_tick_labels': ['$10^{' + str(i * 2 - 4) + '}$' for i in range(6)],
        'v_twin_ticks'      : [10**(i * 2 - 4) for i in range(6)],
        'v_twin_ticks_minor': [10.0**(i * 2 - 3) for i in range(11)],
        'show_legend'       : True,
        'legend_labels'     : [
            '$n_{b} = 10$',
            '$n_{b} = 1000$'
        ],
        'legend_location'   : 'upper left',
        'height'            : 4.0,
        'width'             : 8.0,
        'label_font_size'   : 24,
        'legend_font_size'  : 24,
        'tick_font_size'    : 20
    }
}

# function to calculate the ratio and variance
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
    var = np.min(HLESolver(
        system=system,
        params=params['solver']
    ).get_corr_indices()[:, 0])

    # update results
    return np.array([rat, var], dtype=np.float_)

# function to calculate the ratio and variance
def func_rat_n_beta(system_params):
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

    # get mechanical position and momentum variances
    var_q, var_p = np.min(HLESolver(
        system=system,
        params=params['solver']
    ).get_corr_indices(), axis=0)

    # calculate hyperbolic angles
    r = np.arctanh(rat)
    chr = np.cosh(r)
    shr = np.sinh(r)

    # get phonon number in the Bogoluibov mode
    n_beta = (chr**2 + shr**2) * (var_q + var_p - 1) / 2.0 + shr**2 + chr * shr * (var_q - var_p)

    # update results
    return np.array([rat, n_beta], dtype=np.float_)

if __name__ == '__main__':
    # variance with low thermal phonons
    params['looper']['file_path_prefix'] = 'data/v1.0-qom-v1.0.1/4.4_var_n=10.0'
    params['system']['ns'][1] = 10.0
    looper = run_loopers_in_parallel(
        looper_name='XLooper',
        func=func_rat_var,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    rats, vars_0_rwa = np.transpose(looper.results['V'])
    # variance with high thermal phonons
    params['looper']['file_path_prefix'] = 'data/v1.0-qom-v1.0.1/4.4_var_n=1000.0'
    params['system']['ns'][1] = 1000.0
    looper = run_loopers_in_parallel(
        looper_name='XLooper',
        func=func_rat_var,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    _, vars_0_wrwa = np.transpose(looper.results['V'])

    # occupancy with low thermal phonons
    params['looper']['file_path_prefix'] = 'data/v1.0-qom-v1.0.1/4.4_beta_n=10.0'
    params['system']['ns'][1] = 10.0
    looper = run_loopers_in_parallel(
        looper_name='XLooper',
        func=func_rat_n_beta,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    _, n_betas_0 = np.transpose(looper.results['V'])
    # occupancy with high thermal phonons
    params['looper']['file_path_prefix'] = 'data/v1.0-qom-v1.0.1/4.4_beta_n=1000.0'
    params['system']['ns'][1] = 1000.0
    looper = run_loopers_in_parallel(
        looper_name='XLooper',
        func=func_rat_n_beta,
        params=params['looper'],
        params_system=params['system'],
        plot=False
    )
    _, n_betas_1 = np.transpose(looper.results['V'])

    # plotter
    plotter = MPLPlotter(
        axes={},
        params=params['plotter']
    )
    plotter.update(
        xs=rats,
        vs=- 10 * np.log10([vars_0_rwa, vars_0_wrwa, [0.5] * len(vars_0_rwa)])
    )
    plotter.update_twin_axis(
        vs=[n_betas_0, n_betas_1, [1.0] * len(vars_0_rwa)],
        xs=rats
    )
    plotter.show(
        hold=True
    )