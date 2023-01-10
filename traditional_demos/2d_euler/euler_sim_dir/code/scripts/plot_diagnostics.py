import jax.numpy as np
from jax import jit
import matplotlib.pyplot as plt
import matplotlib as mpl
from generate import convert_representation
from arguments import get_args
import h5py
from poissonsolver import get_poisson_solver
import seaborn as sns

fig, axs = plt.subplots(3, sharex=True, figsize=(4.0,4.65))
lw = 2.5
sns.set_style("whitegrid")
mpl.rc('text', usetex = True)

args = get_args()


nx_exact = 1024
nx = 128
directory = "data/evaldata/EULER_test"

f_exact = h5py.File(
    "{}/vanleer_{}x{}.hdf5".format(directory, nx_exact, nx_exact),
    "r",
)
f_muscl = h5py.File(
    "{}/vanleer_{}x{}.hdf5".format(directory, nx, nx),
    "r",
)
f_gs = h5py.File(
    "{}/gs_{}x{}.hdf5".format(directory, nx, nx),
    "r",
)
f_reduc = h5py.File(
    "{}/gs_reduced_damping_{}x{}.hdf5".format(directory, nx, nx),
    "r",
)

entropy_exact = []
entropy_muscl = []
entropy_gs = []
entropy_reduc = []

energy_exact = []
energy_muscl = []
energy_gs = []
energy_reduc = []

Ts = np.arange(int(args.evaluation_time))

for t in range(int(args.evaluation_time)):

	a_exact = f_exact["a_data"][t]
	a_muscl = f_muscl["a_data"][t]
	a_gs    = f_gs["a_data"][t]
	a_reduc = f_reduc["a_data"][t]

	## entropy

	entropy_exact.append(np.mean(a_exact**2))
	entropy_muscl.append(np.mean(a_muscl**2))
	entropy_gs.append(np.mean(a_gs**2))
	entropy_reduc.append(np.mean(a_reduc**2))

## energy

def get_energy(H):
	nx, ny, _ = H.shape
	dx = args.Lx / nx
	dy = args.Ly / ny
	u_y = -(H[:,:,1] - H[:,:,0]) / dx
	u_x = (H[:,:,3] - H[:,:,0]) / dy
	return np.mean(u_x**2 + u_y**2) * args.Lx * args.Ly



f_poisson_exact = jit(get_poisson_solver(args.poisson_dir, nx_exact, nx_exact, args.Lx, args.Ly, 0))
for t in range(int(args.evaluation_time)):
	a_exact = np.asarray(f_exact["a_data"][t])
	H_exact = f_poisson_exact(a_exact)
	energy_exact.append(get_energy(H_exact))
axs[0].plot(Ts, energy_exact, label="MUSCL {}x{}".format(nx_exact, nx_exact), linewidth=lw, color="#000080",)



f_poisson_solve = jit(get_poisson_solver(args.poisson_dir, nx, nx, args.Lx, args.Ly, 0))
for t in range(int(args.evaluation_time)):
	a_muscl = np.asarray(f_muscl["a_data"][t])
	a_gs    = np.asarray(f_gs["a_data"][t])
	a_reduc = np.asarray(f_reduc["a_data"][t])

	H_muscl = f_poisson_solve(a_muscl)
	H_gs = f_poisson_solve(a_gs)
	H_reduc = f_poisson_solve(a_reduc)

	energy_muscl.append(get_energy(H_muscl))
	energy_gs.append(get_energy(H_gs))
	energy_reduc.append(get_energy(H_reduc))
axs[0].plot(Ts, energy_muscl, label="MUSCL {}x{}".format(nx, nx), linewidth=lw, color="#0F4D92",)
axs[0].plot(Ts, energy_gs, label="GS {}x{} no damping".format(nx, nx), linewidth=lw,color="#0F4D92",linestyle='--')
axs[0].plot(Ts, energy_reduc, label="GS {}x{} reduced damping".format(nx, nx), linewidth=lw,color="#0F4D92",linestyle='dotted')






tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
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
plt.rcParams.update(tex_fonts)



axs[1].plot(Ts, entropy_exact, label="MUSCL {}x{}".format(nx_exact, nx_exact), linewidth=lw, color="#000080")
axs[1].plot(Ts, entropy_muscl, label="MUSCL {}x{}".format(nx, nx), linewidth=lw, color="#0F4D92",)
axs[1].plot(Ts, entropy_gs, label="GS {}x{} no damping".format(nx, nx), linewidth=lw,color="#0F4D92",linestyle='--')
axs[1].plot(Ts, entropy_reduc, label="GS {}x{} reduced damping".format(nx, nx), linewidth=lw,color="#0F4D92",linestyle='dotted')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),loc=(0.13,0.15), prop={'size': 11})

axs[0].set_xlim([0.0,61.0])
axs[0].set_ylim([0.95,1.51])
axs[1].set_ylim([-0.01,0.75])

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

axs[0].set_yticks([1.0, 1.5])
axs[0].set_yticklabels([r'1.0', r'1.5'], usetex=True, fontsize=9)
axs[1].set_yticks([0.5, 0.0])
axs[1].set_yticklabels([r'0.5',r'0'], usetex=True, fontsize=9)



corr_vanleer_128 = []
corr_artificial_stability_128 = []
corr_reduc_75_128 = []

for t in range(int(args.evaluation_time)):

    a_exact = convert_representation(
        f_exact["a_data"][t][None,...], 0, 0, nx, nx, args.Lx, args.Ly, args.equation
    )[0]

    a_muscl = f_muscl["a_data"][t]
    M_muscl = np.concatenate([a_exact[:,:,0].reshape(-1)[:,None], a_muscl[:,:,0].reshape(-1)[:,None]],axis=1)
    corr_vanleer_128.append(np.corrcoef(M_muscl.T)[0,1])

    a_gs = f_gs["a_data"][t]
    M_gs = np.concatenate([a_exact[:,:,0].reshape(-1)[:,None], a_gs[:,:,0].reshape(-1)[:,None]],axis=1)
    corr_artificial_stability_128.append(np.corrcoef(M_gs.T)[0,1])

    a_reduc = f_reduc["a_data"][t]
    M_reduc = np.concatenate([a_exact[:,:,0].reshape(-1)[:,None], a_reduc[:,:,0].reshape(-1)[:,None]],axis=1)
    corr_reduc_75_128.append(np.corrcoef(M_reduc.T)[0,1])


#axs[2].set_xlabel('time', usetex=True,fontsize=12)


axs[2].plot(Ts, np.ones(Ts.shape), color="#000080", label="MUSCL {}x{}".format(nx_exact,nx_exact),linewidth=lw,)
axs[2].plot(Ts, corr_vanleer_128, color="#0F4D92", label="MUSCL 128x128",linewidth=lw,)
axs[2].plot(Ts, corr_artificial_stability_128, linestyle='--', label="GS 128x128 no damping", color="#0F4D92",linewidth=lw,)
axs[2].plot(Ts, corr_reduc_75_128, linestyle='dotted', label="GS 128x128 reduced damping", color="#0F4D92",linewidth=lw,)

axs[2].set_ylim([0.00,1.02])
axs[2].set_xticks([0,30,60.0])
axs[2].set_yticks([0.0,0.5,1.0])
axs[2].set_yticklabels([r'0', r'0.5', r'1.0'], usetex=True, fontsize=9)
axs[2].set_xticklabels([r'$t=0$',r'$t=30$',r'$t=60$'], usetex=True, fontsize=9)
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
axs[2].set_ylabel(' \t\t\t\t\tVorticity correlation', usetex=True, fontsize=12)
axs[1].set_ylabel("Enstrophy", usetex=True, fontsize=12)
axs[0].set_ylabel("Energy", usetex=True, fontsize=12)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

fig.legend(by_label.values(), by_label.keys(),loc=(0.115,0.6), prop={'size': 10})

#fig.subplots_adjust(wspace=0.05, hspace=0.08)
fig.tight_layout()



plt.show()