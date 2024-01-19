from qom.ui.plotters import MPLPlotter

plotter = MPLPlotter(
    axes={},
    params={
        'type'              : 'lines',
        'x_label'           : 'Mechanical Frequency (Hz)',
        'x_scale'           : 'log',
        'x_tick_labels'     : ['$10^{' + str(2 * i) + '}$' for i in range(7)],
        'x_ticks'           : [10**(2 * i) for i in range(7)],
        'x_ticks_minor'     : [10**(i) for i in range(13)],
        'v_label'           : 'Effective Mass (kg)',
        'v_scale'           : 'log',
        'v_tick_labels'     : ['$10^{' + str(-4 * i + 2) + '}$' for i in range(7)],
        'v_ticks'           : [10**(-4 * i + 2) for i in range(7)],
        'v_ticks_minor'     : [10**(-2 * i + 2) for i in range(13)],
        'label_font_size'   : 24,
        'tick_font_size'    : 20,
        'width'             : 9.6,
        'height'            : 9.6
    }
)
plotter.update(
    vs=[1],
    xs=[1]
)
plotter.show()