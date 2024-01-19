# dependencies
import os
import sys

# qom modules
from qom.ui.plotters import MPLPlotter
from qom.utils.loopers import wrap_looper

# add path to local libraries
sys.path.append(os.path.abspath(os.path.join('.')))
# import system
from systems.EndMirror import EM_00

# all parameters
params = {
    'looper': {
        'show_progress' : True,
        'X'             : {
            'var'   : 'Delta_0_norm',
            'min'   : -0.5,
            'max'   : 0.5,
            'dim'   : 100001,
        },
        'Y'             : {
            'var'   : 'g_0_norm',
            'val'   : [0.001, 0.002, 0.004, 0.006]
        }
    },
    'system': {
        'A_l_norm'      : 5.0,
        'Delta_0_norm'  : -1.0, 
        'g_0_norm'      : 0.005,
        'gamma_norm'    : 0.005,
        'kappa_norm'    : 0.15,
        'T_norm'        : 0.0
    },
    'plotter': {
        'type'              : 'scatters',
        'palette'           : 'RdYlBu',
        'bins'              : 21,
        'colors'            : [2, 18, 2, 20, 14, 16, 18, 20],
        'sizes'             : [5] * 8,
        'styles'            : ['s'] * 8,
        'x_label'           : '$\\Delta_{0} / \\omega_{m}$',
        'x_tick_labels'     : ['{:0.1f}'.format(i * 0.1 - 0.4) for i in range(9)],
        'x_ticks'           : [i * 0.1 - 0.4 for i in range(9)],
        'x_ticks_minor'     : [i * 0.05 - 0.4 for i in range(16)],
        'v_label'           : '$N_{o} / 10^{3}$',
        'v_tick_labels'     : list(range(0, 6)),
        'v_ticks'           : list(range(0, 5001, 1000)),
        'v_ticks_minor'     : list(range(0, 5001, 500)),
        'show_legend'       : True,
        'legend_labels'     : [
            '$g_{0} = 0.001 \\omega_{m}$',
            '$g_{0} = 0.002 \\omega_{m}$',
            '$g_{0} = 0.004 \\omega_{m}$',
            '$g_{0} = 0.006 \\omega_{m}$'
        ],
        'legend_range'      : [4, 8],
        'label_font_size'   : 24.0,
        'legend_font_size'  : 20.0,
        'tick_font_size'    : 20.0,
        'width'             : 8.0,
        'height'            : 5.2,
        'vertical_spans'    : [{
            'limits': (-0.325, -0.210),
            'color' : 20,
            'alpha' : 0.2
        }, {
            'limits': (-0.155, -0.145),
            'color' : 18,
            'alpha' : 0.2
        }]
    }
}

# function to obtain the mean optical occupancies
def func_moo(system_params):
    # initialize system
    system = EM_00(
        params=system_params
    )
    # get mean optical occupancies
    N_os = system.get_mean_optical_occupancies()
    
    return N_os

# wrap looper
looper = wrap_looper(
    looper_name='XYLooper',
    func=func_moo,
    params=params['looper'],
    params_system=params['system']
)

# plotter
plotter = MPLPlotter(
    axes={},
    params=params['plotter']
)
V = looper.results['V']
plotter.update(
    vs=[[V[2][j][i] for j in range(len(V[2]))] for i in range(1, 3)] + [[V[3][j][i] for j in range(len(V[3]))] for i in range(1, 3)] + [[V[i][j][0] for j in range(len(V[i]))] for i in range(4)],
    xs=looper.results['X'][0]
)
plotter.show()