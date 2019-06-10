## Yield loci plotter
import matplotlib.pyplot as plt
def in_plane(ifig=None,ft=18):
    import MP.lib
    from MP.lib import mpl_lib
    fig = plt.figure(ifig)
    ax = fig.add_subplot(111)
    mpl_lib.ticks_bins_ax_u(fig.axes,n=3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mathrm{\bar{\Sigma}_{11}}$',
                  dict(fontsize=ft))
    ax.set_ylabel(r'$\mathrm{\bar{\Sigma}_{22}}$',
                  dict(fontsize=ft))
    from MP.lib import mpl_lib
    mpl_lib.tune_x_lim_u(fig.axes)
    return fig
