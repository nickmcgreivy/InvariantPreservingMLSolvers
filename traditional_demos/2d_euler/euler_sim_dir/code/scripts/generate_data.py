from arguments import get_args
from flux import Flux
import jax
import jax.numpy as np
from jax.lax import scan
from jax import jit, vmap
import numpy as onp
import h5py
from functools import partial, reduce
from jax import config
from time import time

config.update("jax_enable_x64", True)

from basisfunctions import num_elements
from rungekutta import FUNCTION_MAP
from training import get_f_phi, get_initial_condition
from flux import Flux
from timederivative import (
    time_derivative_euler
)
from rungekutta import ssp_rk3
from helper import f_to_FV, _evalf_2D_integrate, nabla, convert_representation
from poissonsolver import get_poisson_solver
from poissonbracket import get_poisson_bracket
from diffusion import get_diffusion_func
from helper import legendre_inner_product, inner_prod_with_legendre

PI = np.pi

def create_dataset(args, data_dir, unique_id, nx, ny, nt):
    f = h5py.File(
        "{}/{}_{}x{}.hdf5".format(data_dir, unique_id, nx, ny),
        "w",
    )
    dset_a_new = f.create_dataset(
        "a_data", (nt, nx, ny, 1), dtype="float64"
    )
    f.close()


def write_dataset(args, data_dir, unique_id, a):
    nx, ny = a.shape[1:3]
    f = h5py.File(
        "{}/{}_{}x{}.hdf5".format(data_dir, unique_id, nx, ny),
        "r+",
    )
    dset = f["a_data"]
    dset[:] = a
    f.close()



def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t, x)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t, x)
    return (a_f, t_f), a

def simulate_2D(
    a0,
    t0,
    nx,
    ny,
    Lx,
    Ly,
    dt,
    nt,
    flux,
    output=False,
    f_phi=lambda zeta, t: 0.0,
    f_diffusion=None,
    f_forcing=None,
    f_poisson_bracket=lambda zeta, phi: 0.0,
    rk=ssp_rk3,
    dldt = None,
):
    dx = Lx / nx
    dy = Ly / ny

    dadt = lambda a, t, dldt: time_derivative_euler(
        a,
        t,
        dx,
        dy,
        f_poisson_bracket, 
        f_phi,
        flux,
        dldt = dldt,
    )

    rk_F = lambda a, t, dldt: rk(a, t, dadt, dt, dldt=dldt)

    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), dldt, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), dldt, length=nt)
        return (a_f, t_f)



def generate_eval_data(args, data_dir, a0, nt, dt, flux, dldt = None):

    nx, ny = a0.shape[0:2]

    f_poisson_bracket = get_poisson_bracket(args.poisson_dir, 0, Flux.VANLEER)
    f_poisson_solve = get_poisson_solver(
        args.poisson_dir, nx, ny, args.Lx, args.Ly, 0
    )
    f_phi = lambda zeta, t: f_poisson_solve(zeta)

    @partial(
        jit,
        static_argnums=(
            2
        ),
    )
    def simulate(a_i, t_i, nt, dt):
        return simulate_2D(
            a_i,
            t_i,
            nx,
            ny,
            args.Lx,
            args.Ly,
            dt,
            nt,
            flux,
            output=True,
            f_phi=f_phi,
            f_poisson_bracket=f_poisson_bracket,
            f_diffusion=None,
            f_forcing=None,
            dldt = dldt,
            rk=FUNCTION_MAP[args.runge_kutta],
        )

    return simulate(a0, 0.0, nt, dt)

def generate_exact_data(args, data_dir, a0, nt_one, dt, Tf, flux):
    nx, ny = a0.shape[0:2]
    dx = args.Lx / nx
    dy = args.Ly / ny

    @vmap
    def l2_norm(a):
        """
        a should be (nx, ny, 1), vmapped makes it (_, nx, ny, 1)
        """
        return 1/2 * np.sum(a**2) * dx * dy

    data_list = []
    dldt_list = []
    for t in range(int(Tf)):
        print(t)
        a_data = generate_eval_data(args, data_dir, a0, nt_one + 1, dt, flux)
        dldt_out = (l2_norm(a_data[1:]) - l2_norm(a_data[:-1])) / dt
        
        data_list.append(a0)
        dldt_list.append(dldt_out)
        a0 = a_data[-1]

    a_data = np.asarray(data_list)
    dldt_out = np.asarray(dldt_list).reshape(-1)

    return a_data, dldt_out


def main():
    args = get_args()

    nx = 128
    ny = nx
    UPSAMPLE = 4
    nx_exact = nx * UPSAMPLE
    ny_exact = nx_exact
    Tf = 60.0
    data_dir = args.eval_dir
    unique_ids = ["exact", "vanleer", "gs0", "gs_dldtexact", "ec", "ec_dldtexact"]


    #### generate initial condition
    key = jax.random.PRNGKey(args.random_seed)
    f_init = get_initial_condition(key, args)
    t0 = 0.0
    a0_exact = f_to_FV(nx_exact, ny_exact, args.Lx, args.Ly, 0, f_init, t0, n = 8)
    a0 = convert_representation(a0_exact[None], 0, 0, nx, ny, args.Lx, args.Ly)[0]


    #### generate timesteps
    dx = args.Lx / nx
    dy = args.Ly / ny
    dt = args.cfl_safety * ((dx * dy) / (dx + dy)) # initial dt based on cfl
    nt_one = int(1.0 / dt) + 1
    dt = 1.0 / nt_one # final dt which gives exactly t = 1.0 after nt_one steps 
    nt = int(Tf / dt)
    dt_exact = dt / UPSAMPLE
    nt_exact = nt * UPSAMPLE
    nt_one_exact = nt_one * UPSAMPLE
    flux_exact = Flux.VANLEER

    ### create datasets
    create_dataset(args, data_dir, unique_ids[0], nx_exact, ny_exact, int(Tf))
    for j in range(1, len(unique_ids)):
        create_dataset(args, data_dir, unique_ids[j], nx, ny, int(Tf))



    #### generate exact data at intervals
    exact_data, dldt_exact = generate_exact_data(args, data_dir, a0_exact, nt_one_exact, dt_exact, Tf, flux_exact)

    #### store exact data at int(Tf) intervals
    exact_data_ds = exact_data#[::nt_one*UPSAMPLE]
    dldt_exact_ds = dldt_exact[::UPSAMPLE]
    write_dataset(args, data_dir, unique_ids[0], exact_data_ds)



    #### create data for low-resolution simulations
    vanleer_data = generate_eval_data(args, data_dir, a0, nt, dt, Flux.VANLEER)
    conservation_data = generate_eval_data(args, data_dir, a0, nt, dt, Flux.CONSERVATION, dldt = np.zeros(dldt_exact_ds.shape))
    conservation_dldtexact_data = generate_eval_data(args, data_dir, a0, nt, dt, Flux.CONSERVATION, dldt = dldt_exact_ds)
    energyconservation_data = generate_eval_data(args, data_dir, a0, nt, dt, Flux.ENERGYCONSERVATION)
    energyconservation_dldtexact_data = generate_eval_data(args, data_dir, a0, nt, dt, Flux.ENERGYCONSERVATION, dldt = dldt_exact_ds)

    #### store data for low-resolution simulations
    vanleer_data_ds = vanleer_data[::nt_one]

    write_dataset(args, data_dir, unique_ids[1], vanleer_data_ds)
    conservation_data_ds = conservation_data[::nt_one]
    write_dataset(args, data_dir, unique_ids[2], conservation_data_ds)
    conservation_dldtexact_data_ds = conservation_dldtexact_data[::nt_one]
    write_dataset(args, data_dir, unique_ids[3], conservation_dldtexact_data_ds)
    energyconservation_data_ds = energyconservation_data[::nt_one]
    write_dataset(args, data_dir, unique_ids[4], energyconservation_data_ds)
    energyconservation_dldtexact_data_ds = energyconservation_dldtexact_data[::nt_one]
    write_dataset(args, data_dir, unique_ids[5], energyconservation_dldtexact_data_ds)
    
    

if __name__ == "__main__":
    main()
