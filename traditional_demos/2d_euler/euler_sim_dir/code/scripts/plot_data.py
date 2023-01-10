import matplotlib.pyplot as plt
import jax
import numpy as np
import h5py
from functools import lru_cache
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import seaborn as sns
import matplotlib as mpl

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
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
plt.rcParams.update(tex_fonts)


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
    fig, axs = plt.subplots(
        3,
        4,
        figsize=(6.5, 4.65),
        squeeze=False,
        sharex=True,
        sharey=True,
        #constrained_layout=True,
    )

    ts = [0.0, 0.5, 1.0]
    Lx = Ly = 2 * np.pi
    vmin = -2
    vmax = 2
    nx = 128
    nx_exact = 1024

    for j, t in enumerate(ts):
        
        f_exact = h5py.File(
            "data/evaldata/EULER_test/vanleer_{}x{}.hdf5".format(nx_exact, nx_exact),
            "r",
        )
        f_muscl = h5py.File(
            "data/evaldata/EULER_test/vanleer_{}x{}.hdf5".format(nx, nx),
            "r",
        )
        f_gs = h5py.File(
            "data/evaldata/EULER_test/gs_{}x{}.hdf5".format(nx, nx),
            "r",
        )
        f_reduc = h5py.File(
            "data/evaldata/EULER_test/gs_reduced_damping_{}x{}.hdf5".format(nx, nx),
            "r",
        )
        index = int(t * (f_exact["a_data"].shape[0] - 1))
        zeta_exact = f_exact["a_data"][index]
        index = int(t * (f_muscl["a_data"].shape[0] - 1))
        zeta_muscl = f_muscl["a_data"][index]
        zeta_gs = f_gs["a_data"][index]
        zeta_reduc = f_reduc["a_data"][index]


        im_exact = plot_axs(
            average_half(average_half(average_half(zeta_exact))),
            axs[j,0],
            Lx,
            Ly,
            vmin,
            vmax,
            "MUSCL 1024x1024",
        )
        im_muscl = plot_axs(
            average_half(zeta_muscl),
            axs[j,1],
            Lx,
            Ly,
            vmin,
            vmax,
            "MUSCL 128x128",
        )
        im_gs = plot_axs(
            average_half(zeta_gs),
            axs[j,2],
            Lx,
            Ly,
            vmin,
            vmax,
            "GS 128x128\nno damping",
        )
        im_reduc = plot_axs(
            average_half(zeta_reduc),
            axs[j,3],
            Lx,
            Ly,
            vmin,
            vmax,
            "GS 128x128\n reduced damping",
        )

    for j, t in enumerate(ts):
        axs[j, 0].set_ylabel("t = {}".format(int(t * 60)))

    axs[0,0].set_title("MUSCL\n1024x1024")
    axs[0,1].set_title("MUSCL\n128x128")
    axs[0,2].set_title("GS 128x128\nno damping")
    axs[0,3].set_title("GS 128x128\nreduced damping")

    axs[2,-1].yaxis.set_label_position("right")
    axs[2,-1].set_ylabel("'\n ")

    for i in range(3):
        for j in range(4):
            axs[i, j].set_xlim([0, Lx])
            axs[i, j].set_ylim([0, Ly])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i,j].set(aspect='equal')
        
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    cbar_ax = fig.add_axes([0.915, 0.06, 0.03, 0.8])
    fig.colorbar(im_reduc, cax=cbar_ax,ticks=[vmin, 0, vmax])
    

plot_data()
#plt.savefig('vorticity_plots.eps')
plt.show()
