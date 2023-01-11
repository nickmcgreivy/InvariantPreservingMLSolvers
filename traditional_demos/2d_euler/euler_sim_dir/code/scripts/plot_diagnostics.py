import jax.numpy as np
from jax import jit, vmap
import matplotlib.pyplot as plt
import matplotlib as mpl
from helper import convert_representation
from arguments import get_args
import h5py
from poissonsolver import get_poisson_solver
import seaborn as sns
from jax import config
config.update("jax_enable_x64", True)


fig, axs = plt.subplots(3, sharex=True, figsize=(5.0,6.75))
lw = 2.5
usetex = False
sns.set_style("whitegrid")
mpl.rc('text')
fonts = {
    # Use LaTeX to write all text
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.style.use('seaborn')
plt.rcParams.update(fonts)



args = get_args()
nx = 128
UPSAMPLE = 4
nx_exact = nx * UPSAMPLE
unique_ids = ["exact", "vanleer", "gs0", "gs_dldtexact", "ec", "ec_dldtexact"]
labels = [
"MUSCL\n{}x{}".format(nx_exact, nx_exact), 
"MUSCL",
r'$\frac{d\ell_{2}^{new}}{dt} = 0$', #\n\frac{d\ell_2^{\textnormal{new}}}{dt} = 0
r'$\frac{d\ell_{2}^{new}}{dt} = \frac{d\ell_{2}^{exact}}{dt}$', #\n\frac{d\ell_2^{\textnormal{new}}}{dt} = \frac{d\ell_2^{\textnormal{exact}}}{dt}
r'EC $\frac{d\ell_{2}^{new}}{dt} = \frac{d\ell_{2}^{old}}{dt}$', #\n\frac{d\ell_2^{\textnormal{new}}}{dt} = \frac{d\ell_2^{\textnormal{old}}}{dt}
r"EC $\frac{d\ell_{2}^{new}}{dt} = \frac{d\ell_{2}^{exact}}{dt}$", # \n\frac{d\ell_2^{\textnormal{new}}}{dt} = \frac{d\ell_2^{\textnormal{exact}}}{dt}
]
nxs = [nx_exact, nx, nx, nx, nx, nx]
N = len(unique_ids)
directory = "data/evaldata/EULER_test"

colors = ["#008000", "#000080", "#0F4D92", "#0F4D92", "#924D0F", "#924D0F"]
linestyles = [None, None, '--', 'dotted', None, 'dotted']

@vmap
def get_enstrophy(a):
    return 0.5 * np.mean(a**2)

@vmap
def get_energy(H):
    nx, ny, _ = H.shape
    dx = args.Lx / nx
    dy = args.Ly / ny
    u_y = -(H[:,:,1] - H[:,:,0]) / dx
    u_x = (H[:,:,3] - H[:,:,0]) / dy
    return np.mean(u_x**2 + u_y**2) * args.Lx * args.Ly


f_exact = h5py.File(
"{}/{}_{}x{}.hdf5".format(directory, unique_ids[0], nxs[0], nxs[0]),
"r",
)
a_exact = f_exact["a_data"][:]
a_exact_ds = convert_representation(a_exact, 0, 0, nx, nx, args.Lx, args.Ly)
f_exact.close()


for k in range(N):
    f = h5py.File(
    "{}/{}_{}x{}.hdf5".format(directory, unique_ids[k], nxs[k], nxs[k]),
    "r",
    )

    a_data = f["a_data"][:]
    f.close()

    ### enstrophy
    enstrophy = get_enstrophy(a_data)

    ### energy
    f_poisson = vmap(jit(get_poisson_solver(args.poisson_dir, nxs[k], nxs[k], args.Lx, args.Ly, 0)))
    H = f_poisson(a_data)
    energy = get_energy(H)


    ### correlation
    nt = a_data.shape[0]
    if k == 0:
        a_data_ds = convert_representation(a_data, 0, 0, nx, nx, args.Lx, args.Ly)
        M = np.concatenate([a_exact_ds[...,0].reshape(nt, -1)[...,None], a_data_ds[...,0].reshape(nt,-1)[...,None]],axis=-1) 
    else:
        M = np.concatenate([a_exact_ds[...,0].reshape(nt, -1)[...,None], a_data[...,0].reshape(nt,-1)[...,None]],axis=-1)
    corr = vmap(np.corrcoef)(np.swapaxes(M,1,2))[:,0,1]



    Ts = np.arange(nt)
    #### plot
    axs[0].plot(Ts, energy, label=labels[k], linewidth=lw, color=colors[k], linestyle=linestyles[k])
    axs[1].plot(Ts, enstrophy, label=labels[k], linewidth=lw, color=colors[k], linestyle=linestyles[k])
    axs[2].plot(Ts, corr, label=labels[k], linewidth=lw, color=colors[k], linestyle=linestyles[k])


axs[0].set_xlim([0.0,61.0])
axs[0].set_ylim([0.95,1.35])
axs[1].set_ylim([-0.01,0.375])

axs[0].grid(visible=False)
axs[0].set_facecolor('white')
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['left'].set_visible(True)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

axs[1].grid(visible=False)
axs[1].set_facecolor('white')
axs[1].spines['bottom'].set_visible(False)
axs[1].spines['left'].set_visible(True)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)


axs[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
axs[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

axs[1].set_xticks([])
axs[0].set_xticks([])
axs[1].set_xticklabels([])
axs[0].set_xticklabels([])

axs[0].set_yticks([1.0, 1.25])
axs[0].set_yticklabels([r'1.0', r'1.25'], usetex=usetex, fontsize=11)
axs[1].set_yticks([0.25, 0.0])
axs[1].set_yticklabels([r'0.25',r'0'], usetex=usetex, fontsize=11)


axs[2].set_ylim([0.00,1.02])
axs[2].set_xticks([0,30,60.0])
axs[2].set_yticks([0.0,0.5,1.0])
axs[2].set_yticklabels([r'0', r'0.5', r'1.0'], usetex=usetex, fontsize=11)
axs[2].set_xticklabels([r'$t=0$',r'$t=30$',r'$t=60$'], usetex=usetex, fontsize=11)
for i, tick in enumerate(axs[2].xaxis.get_majorticklabels()):
    if i == 0:
        tick.set_horizontalalignment("left")
    if i == 2:
        tick.set_horizontalalignment("right")

axs[2].grid(visible=False)
axs[2].set_facecolor('white')
axs[2].spines['bottom'].set_visible(True)
axs[2].spines['left'].set_visible(True)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)


axs[0].yaxis.set_label_position("right")
axs[1].yaxis.set_label_position("right")
axs[2].yaxis.set_label_position("right")
axs[2].set_ylabel('Vorticity correlation', usetex=usetex, fontsize=14)
axs[1].set_ylabel("Enstrophy", usetex=usetex, fontsize=14)
axs[0].set_ylabel("Energy", usetex=usetex, fontsize=14)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
vals = list(by_label.values())
keys = list(by_label.keys())
fig.legend(vals, keys,loc=(0.13,0.595), prop={'size': 13}, ncol=2)
#fig.legend(vals[0:2], keys[0:2],loc=(0.13,0.65), prop={'size': 11}, ncol=2)
#fig.legend(vals[2:4], keys[2:4],loc=(0.13,0.55), prop={'size': 11}, ncol=2)
#fig.legend(vals[4:], keys[4:],  loc=(0.13,0.35), prop={'size': 11}, ncol=2)

#fig.subplots_adjust(wspace=0.05, hspace=0.08)

fig.tight_layout()

#plt.savefig('euler_diagnostics.eps')
#plt.savefig('euler_diagnostics.png')

plt.show()
