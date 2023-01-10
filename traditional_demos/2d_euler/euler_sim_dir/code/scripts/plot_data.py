import matplotlib.pyplot as plt
import jax
import numpy as np
import h5py
from functools import lru_cache
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import seaborn as sns
import matplotlib as mpl

fonts = {
    # Use LaTeX to write all text
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.style.use('seaborn')
plt.rcParams.update(fonts)


def plot_axs(
    zeta, axs, Lx, Ly, vmin, vmax, label, plotting_density=4, cmap=sns.cm.icefire,
):
    nx, ny, num_elem = zeta.shape

    output = np.zeros((nx, ny))

    output = np.sum(zeta, axis=-1)

    x_plot = np.linspace(0, Lx, nx + 1)
    y_plot = np.linspace(0, Ly, ny + 1)

    return axs.pcolormesh(
        x_plot,
        y_plot,
        output.T,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label=label,
    )


def average_half(zeta):
    return (zeta[::2,::2] + zeta[1::2,1::2] + zeta[1::2, ::2,] + zeta[::2,1::2]) / 4


def plot_data():
    nx = 128
    UPSAMPLE = 2
    nx_exact = nx * UPSAMPLE
    unique_ids = ["exact", "vanleer", "gs0", "gs_dldtexact", "ec", "ec_dldtexact"]
    labels = [
    "MUSCL\n{}x{}".format(nx_exact, nx_exact), 
    "MUSCL\n{}x{}".format(nx, nx),
    "ENC {}x{}\nzero damping".format(nx, nx), # \n\frac{d\ell_2^{\textnormal{new}}}{dt} = 0
    "ENC {}x{}\nexact damping".format(nx, nx), # \n\frac{d\ell_2^{\textnormal{new}}}{dt} = \frac{d\ell_2^{\textnormal{exact}}}{dt}
    "EC {}x{}\nnormal damping".format(nx, nx), # \n\frac{d\ell_2^{\textnormal{new}}}{dt} = \frac{d\ell_2^{\textnormal{old}}}{dt}
    "EC {}x{}\nexact damping".format(nx, nx), # \n\frac{d\ell_2^{\textnormal{new}}}{dt} = \frac{d\ell_2^{\textnormal{exact}}}{dt}
    ]
    
    N = len(unique_ids)

    fig, axs = plt.subplots(
        3,
        N,
        figsize=(10.0, 5.0),
        squeeze=False,
        sharex=True,
        sharey=True,
        #constrained_layout=True,
    )

    ts = [0.0, 0.5, 1.0]
    Lx = Ly = 2 * np.pi
    vmin = -2
    vmax = 2

    for j, t in enumerate(ts):

        f_exact = h5py.File(
            "data/evaldata/EULER_test/{}_{}x{}.hdf5".format(unique_ids[0], nx_exact, nx_exact),
            "r",
        )
        index = int(t * (f_exact["a_data"].shape[0] - 1))
        zeta_exact = f_exact["a_data"][index]
        plot_axs(
            average_half(zeta_exact),
            axs[j,0],
            Lx,
            Ly,
            vmin,
            vmax,
            labels[0],
        )
        f_exact.close()
        for k in range(1, N):
            unique_id = unique_ids[k]
            f = h5py.File(
                "data/evaldata/EULER_test/{}_{}x{}.hdf5".format(unique_id, nx, nx),
                "r",
            )
            index = int(t * (f["a_data"].shape[0] - 1))
            zeta = f["a_data"][index]


            im = plot_axs(
                zeta,
                axs[j,k],
                Lx,
                Ly,
                vmin,
                vmax,
                labels[k],
            )
            f.close()

    for j, t in enumerate(ts):
        axs[j, 0].set_ylabel("t = {}".format(int(t * 60)))

    for k in range(N):
        axs[0,k].set_title(labels[k])

    axs[2,-1].yaxis.set_label_position("right")
    axs[2,-1].set_ylabel("'\n ")

    for i in range(3):
        for j in range(N):
            axs[i, j].set_xlim([0, Lx])
            axs[i, j].set_ylim([0, Ly])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set(aspect='equal')
        
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    cbar_ax = fig.add_axes([0.943, 0.06, 0.03, 0.8])
    fig.colorbar(im, cax=cbar_ax,ticks=[vmin, 0, vmax])
    

plot_data()
#plt.savefig('vorticity_plots.eps')
plt.show()
