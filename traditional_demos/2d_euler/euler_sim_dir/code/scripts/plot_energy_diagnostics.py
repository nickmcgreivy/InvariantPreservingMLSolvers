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
f_gs_ec = h5py.File(
    "{}/gs_ec_{}x{}.hdf5".format(directory, nx, nx),
    "r",
)
f_gs_ec_reduc = h5py.File(
    "{}/gs_ec_reduced_damping_{}x{}.hdf5".format(directory, nx, nx),
    "r",
)

entropy_exact = []
entropy_muscl = []
entropy_gs_ec = []
entropy_gs_ec_reduc = []

energy_exact = []
energy_muscl = []
energy_gs_ec = []
energy_gs_ec_reduc = []

Ts = np.arange(int(args.evaluation_time))

for t in range(int(args.evaluation_time)):

	a_exact = f_exact["a_data"][t]
	a_muscl = f_muscl["a_data"][t]
	a_gs_ec    = f_gs_ec["a_data"][t]
	a_gs_ec_reduc = f_gs_ec_reduc["a_data"][t]

	## entropy

	entropy_exact.append(np.mean(a_exact**2))
	entropy_muscl.append(np.mean(a_muscl**2))
	entropy_gs_ec.append(np.mean(a_gs_ec**2))
	entropy_gs_ec_reduc.append(np.mean(a_gs_ec_reduc**2))

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
	a_gs_ec    = np.asarray(f_gs_ec["a_data"][t])
	a_gs_ec_reduc = np.asarray(f_gs_ec_reduc["a_data"][t])

	H_muscl = f_poisson_solve(a_muscl)
	H_gs_ec = f_poisson_solve(a_gs_ec)
	H_gs_ec_reduc = f_poisson_solve(a_gs_ec_reduc)

	energy_muscl.append(get_energy(H_muscl))
	energy_gs_ec.append(get_energy(H_gs_ec))
	energy_gs_ec_reduc.append(get_energy(H_gs_ec_reduc))
axs[0].plot(Ts, energy_muscl, label="MUSCL {}x{}".format(nx, nx), linewidth=lw, color="#0F4D92",)
axs[0].plot(Ts, energy_gs_ec, label="GS EC {}x{} normal damping".format(nx, nx), linewidth=lw,color="#ff5555",linestyle='--')
axs[0].plot(Ts, energy_gs_ec_reduc, label="GC EC {}x{} no damping".format(nx, nx), linewidth=lw,color="#ff5555",linestyle='dotted')






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
axs[1].plot(Ts, entropy_gs_ec, label="GC EC {}x{} normal damping".format(nx, nx), linewidth=lw,color="#ff5555",linestyle='--')
axs[1].plot(Ts, entropy_gs_ec_reduc, label="GC EC {}x{} no damping".format(nx, nx), linewidth=lw,color="#ff5555",linestyle='dotted')


axs[0].set_xlim([0.0,61.0])
axs[0].set_ylim([0.99,1.27])
axs[1].set_ylim([-0.01,1.25])

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
axs[0].set_yticklabels([r'1.0', r'1.25'], usetex=True, fontsize=9)
axs[1].set_yticks([1.0, 0.5, 0.0])
axs[1].set_yticklabels([r'1.0', r'0.5',r'0'], usetex=True, fontsize=9)



corr_vanleer_128 = []
corr_gs_ec_128 = []
corr_gs_ec_reduc_128 = []

for t in range(int(args.evaluation_time)):

    a_exact = convert_representation(
        f_exact["a_data"][t][None,...], 0, 0, nx, nx, args.Lx, args.Ly, args.equation
    )[0]

    a_muscl = f_muscl["a_data"][t]
    M_muscl = np.concatenate([a_exact[:,:,0].reshape(-1)[:,None], a_muscl[:,:,0].reshape(-1)[:,None]],axis=1)
    corr_vanleer_128.append(np.corrcoef(M_muscl.T)[0,1])

    a_gs_ec = f_gs_ec["a_data"][t]
    M_gs_ec = np.concatenate([a_exact[:,:,0].reshape(-1)[:,None], a_gs_ec[:,:,0].reshape(-1)[:,None]],axis=1)
    corr_gs_ec_128.append(np.corrcoef(M_gs_ec.T)[0,1])

    a_gs_ec_reduc = f_gs_ec_reduc["a_data"][t]
    M_gs_ec_reduc = np.concatenate([a_exact[:,:,0].reshape(-1)[:,None], a_gs_ec_reduc[:,:,0].reshape(-1)[:,None]],axis=1)
    corr_gs_ec_reduc_128.append(np.corrcoef(M_gs_ec_reduc.T)[0,1])


#axs[2].set_xlabel('time', usetex=True,fontsize=12)


axs[2].plot(Ts, np.ones(Ts.shape), color="#000080", label="MUSCL {}x{}".format(nx_exact,nx_exact),linewidth=lw,)
axs[2].plot(Ts, corr_vanleer_128, color="#0F4D92", label="MUSCL 128x128",linewidth=lw,)
axs[2].plot(Ts, corr_gs_ec_128, linestyle='--', label="GS EC 128x128 normal damping", color="#ff5555",linewidth=lw,)
axs[2].plot(Ts, corr_gs_ec_reduc_128, linestyle='dotted', label="GS EC 128x128 no damping", color="#ff5555",linewidth=lw,)

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

fig.legend(by_label.values(), by_label.keys(),loc=(0.15,0.63), prop={'size': 10})

#fig.subplots_adjust(wspace=0.05, hspace=0.08)
fig.tight_layout()



plt.show()