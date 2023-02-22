"""
Generates code for solving the 2D poisson equation

nabla^2 phi = - zeta

on a rectangular domain (x,y) in ([0, L_x], [0, L_y])

where phi is represented by a continous FE basis
and zeta is represented by a discontinous Legendre basis

The solution phi_j is given by the weak form

(B_{ij} - V_{ij}) phi_j = S_{ij} zeta_j

The continuous solution phi is given by

phi = sum_j phi_j psi_j(x,y)

where psi_j(x,y) are the FE basis functions.
The sum runs over every element in the simulation.
The discontinuous input zeta is given

zeta_k = sum_j zeta_kj alpha_j(x_k,y_k)

where alpha_j are the legendre basis functions.
"""
import sympy
import numpy as np
from sympy.matrices import Matrix, zeros
from sympy import Rational, symbols, integrate, nsimplify, diff, lambdify
from functools import lru_cache, partial
import os.path
import os
import jax.numpy as jnp
import jax
from jax.experimental import sparse as jsparse
from jax.experimental.sparse.linalg import spsolve
from jax.lib import xla_bridge
from scipy import sparse
from scipy.sparse import dok_matrix
from jax.scipy.linalg import lu_factor
import timeit
import time
from jax import config, jit
import jax.experimental.host_callback as hcb


config.update("jax_enable_x64", True)

from basisfunctions import (
    legendre_poly,
    FE_poly,
    node_locations,
    num_elements,
    num_elements_FE,
)
import sparsesolve


def get_bottom_indices(order):
    if order == 1 or order == 0:
        return jnp.asarray([0], dtype=int)
    raise Exception


def is_bottom_element(order, k):
    arr = get_bottom_indices(order)
    if order == 1 or order == 0:
        if k in arr:
            return True
    else:
        raise Exception
    return False


def convert_to_bottom_indices(T, order):
    def convert_to_bottom_index(index):
        if order == 1 or order == 0:
            if index == 0:
                return 0
            else:
                raise Exception

    T = np.asarray(T, dtype=int)
    T_new = np.zeros(T.shape)
    T_new[:, 0] = T[:, 0]
    T_new[:, 1] = T[:, 1]
    for i in range(T.shape[0]):
        T_new[i, 2] = convert_to_bottom_index(T[i, 2])
    return jnp.asarray(T_new, dtype=int)


def load_assembly_matrix(basedir, nx, ny, order):
    def create_assembly_matrix(nx, ny, order):
        """
        Generates an assembly matrix which converts the
        local/element matrices to the global matrices
        """
        table = {}
        nodes = node_locations(order)
        num_elem = nodes.shape[0]

        def lookup_table(ijk):
            i, j, k = ijk
            x, y = nodes[k, :]
            i_l = (i - 1) % nx
            i_r = (i + 1) % nx
            j_b = (j - 1) % ny
            j_t = (j + 1) % ny
            if (i, j, x, y) in table:
                return table[(i, j, x, y)]
            elif (i_l, j, x + 2, y) in table:
                return table[(i_l, j, x + 2, y)]
            elif (i_r, j, x - 2, y) in table:
                return table[(i_r, j, x - 2, y)]
            elif (i, j_t, x, y - 2) in table:
                return table[(i, j_t, x, y - 2)]
            elif (i, j_b, x, y + 2) in table:
                return table[(i, j_b, x, y + 2)]
            elif (i_l, j_t, x + 2, y - 2) in table:
                return table[(i_l, j_t, x + 2, y - 2)]
            elif (i_r, j_t, x - 2, y - 2) in table:
                return table[(i_r, j_t, x - 2, y - 2)]
            elif (i_l, j_b, x + 2, y + 2) in table:
                return table[(i_l, j_b, x + 2, y + 2)]
            elif (i_r, j_b, x - 2, y + 2) in table:
                return table[(i_r, j_b, x - 2, y + 2)]
            else:
                return None

        def assign_table(ijk, node_val):
            i, j, k = ijk
            x, y = nodes[k, :]
            table[(i, j, x, y)] = node_val
            return

        node_index = 0
        for j in range(ny):
            for i in range(nx):
                for k in range(num_elem):
                    ijk = (i, j, k)
                    node_val = lookup_table(ijk)
                    if node_val is None:
                        node_val = node_index
                        node_index += 1
                    assign_table(ijk, node_val)

        num_global_elements = max(table.values()) + 1
        M = np.zeros((nx, ny, num_elem), dtype=int)
        T = -np.ones((num_global_elements, 3), dtype=int)

        for i in range(nx):
            for j in range(ny):
                for k in range(num_elem):
                    x, y = nodes[k, :]
                    gamma = table[(i, j, x, y)]
                    M[i, j, k] = gamma
                    if T[gamma, 0] == -1 and is_bottom_element(order, k):
                        T[gamma, 0] = i
                        T[gamma, 1] = j
                        T[gamma, 2] = k

        return num_global_elements, M, T

    if os.path.exists(
        "{}/data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
            basedir, nx, ny, order
        )
    ):
        num_global_elements = np.load(
            "{}/data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
                basedir, nx, ny, order
            )
        )
        M = np.load(
            "{}/data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
                basedir, nx, ny, order
            )
        )
        T = np.load(
            "{}/data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
                basedir, nx, ny, order
            )
        )
    else:
        num_global_elements, M, T = create_assembly_matrix(nx, ny, order)
        np.save(
            "{}/data/poissonmatrices/num_global_elements_nx{}_ny{}_order{}.npy".format(
                basedir, nx, ny, order
            ),
            num_global_elements,
        )
        np.save(
            "{}/data/poissonmatrices/assembly_matrix_nx{}_ny{}_order{}.npy".format(
                basedir, nx, ny, order
            ),
            M,
        )
        np.save(
            "{}/data/poissonmatrices/assembly_matrix_transpose_nx{}_ny{}_order{}.npy".format(
                basedir, nx, ny, order
            ),
            T,
        )
    return num_global_elements, M, T


def load_elementwise_volume(basedir, nx, ny, Lx, Ly, order):
    """
    Returns the (num_elements x num_elements) matrix
    where the (i,j) component is the elementwise integral
    V_{ij} = int_Omega nabla psi_i nabla psi_j dx dy
    in "local" coordinates.

    Later we will map this matrix to "global" coordinates.
    """

    def create_elementwise_volume(order):
        basis = FE_poly(order)
        num_elem = basis.shape[0]
        res1 = np.zeros((num_elem, num_elem))
        res2 = np.zeros((num_elem, num_elem))
        for i in range(num_elem):
            for j in range(num_elem):
                expr1 = diff(basis[i], "x") * diff(basis[j], "x")
                res1[i, j] = integrate(expr1, ("x", -1, 1), ("y", -1, 1))
                expr2 = diff(basis[i], "y") * diff(basis[j], "y")
                res2[i, j] = integrate(expr2, ("x", -1, 1), ("y", -1, 1))
        return res1, res2

    dx = Lx / nx
    dy = Ly / ny
    if os.path.exists(
        "{}/data/poissonmatrices/elementwise_volume_{}_1.npy".format(basedir, order)
    ):
        res1 = np.load(
            "{}/data/poissonmatrices/elementwise_volume_{}_1.npy".format(basedir, order)
        )
        res2 = np.load(
            "{}/data/poissonmatrices/elementwise_volume_{}_2.npy".format(basedir, order)
        )
    else:
        res1, res2 = create_elementwise_volume(order)
        np.save(
            "{}/data/poissonmatrices/elementwise_volume_{}_1".format(basedir, order),
            res1,
        )
        np.save(
            "{}/data/poissonmatrices/elementwise_volume_{}_2".format(basedir, order),
            res2,
        )
    V = res1 * (dy / dx) + res2 * (dx / dy)
    return V


def load_elementwise_source(basedir, nx, ny, Lx, Ly, order):
    def write_elementwise_source(order):
        FE_basis = FE_poly(order)
        legendre_basis = legendre_poly(order)
        res = np.zeros((FE_basis.shape[0], legendre_basis.shape[0]))
        for i in range(FE_basis.shape[0]):
            for j in range(legendre_basis.shape[0]):
                expr = FE_basis[i] * legendre_basis[j]
                res[i, j] = integrate(expr, ("x", -1, 1), ("y", -1, 1))
        return res

    dx = Lx / nx
    dy = Ly / ny
    if os.path.exists(
        "{}/data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order)
    ):
        res = np.load(
            "{}/data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order)
        )
    else:
        res = write_elementwise_source(order)
        np.save(
            "{}/data/poissonmatrices/elementwise_source_{}.npy".format(basedir, order),
            res,
        )
    return res * dx * dy / 4


def load_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements):
    if os.path.exists(
        "{}/data/poissonmatrices/volume_{}_{}_{}.npz".format(basedir, nx, ny, order)
    ):
        sV = sparse.load_npz(
            "{}/data/poissonmatrices/volume_{}_{}_{}.npz".format(basedir, nx, ny, order)
        )
    else:
        V = create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements)
        sV = sparse.csr_matrix(V)
        sparse.save_npz(
            "{}/data/poissonmatrices/volume_{}_{}_{}.npz".format(
                basedir, nx, ny, order
            ),
            sV,
        )
    return sV


def create_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, num_global_elements):
    num_elem = num_elements(order)
    K_elementwise = load_elementwise_volume(basedir, nx, ny, Lx, Ly, order)

    sK = dok_matrix((num_global_elements, num_global_elements))

    for j in range(ny):
        for i in range(nx):
            sK[M[i, j, :][:, None], M[i, j, :][None, :]] += K_elementwise[:, :]
    return sK

    
def get_kernel(order):
    bottom_indices = get_bottom_indices(order)
    K = np.zeros((2, 2, num_elements_FE(order), num_elements_FE(order)))
    if order == 1 or order == 0:
        K[0, 0, 0, 2] = 1.0
        K[1, 0, 0, 3] = 1.0
        K[0, 1, 0, 1] = 1.0
        K[1, 1, 0, 0] = 1.0
    else:
        raise Exception
    return jnp.asarray(K)[:, :, bottom_indices, :]


######
# Poisson solver
######

def get_poisson_solve_fn_fv(sim_params):
    basedir = sim_params.basedir
    nx = sim_params.nx
    ny = sim_params.ny
    order = 0
    Lx = sim_params.Lx
    Ly = sim_params.Ly
    N_global_elements, M, T = load_assembly_matrix(basedir, nx, ny, order)
    T = convert_to_bottom_indices(T, order)
    S_elem = load_elementwise_source(basedir, nx, ny, Lx, Ly, order)
    K = get_kernel(order) @ S_elem


    sV = load_volume_matrix(basedir, nx, ny, Lx, Ly, order, M, N_global_elements)
    V_sp = jsparse.BCOO.from_scipy_sparse(sV)
    args = V_sp.data, V_sp.indices, N_global_elements
    kwargs = {"forward": True}
    #tol = 1e-8
    #V = sparse.csr_matrix.todense(sV)[1:, 1:]
    #data, indices, indptr = jsparse.csr_fromdense(V, nse=np.count_nonzero(V))
    #jax_lu_solve = lambda b: jsparse.sparsify(spsolve)(data, indices, indptr, b, tol=tol)

    platform = xla_bridge.get_backend().platform

    if platform == 'cpu':
        custom_lu_solve = jax.jit(lambda b: sparsesolve.sparse_solve_prim(b, *args, **kwargs))

        def solve_step(b):
            res = custom_lu_solve(b)
            return res - jnp.mean(res)

    elif platform == 'gpu':
        custom_lu_solve = jax.jit(lambda b: sparsesolve.sparse_solve_prim(b, *args, **kwargs), backend='cpu')
        def solve_step(b):
            res = jax.pure_callback(custom_lu_solve, b, b)
            return res - jnp.mean(res)

    else:
        raise Exception

    def solve(xi):
        xi = xi - jnp.mean(xi)
        xi = jnp.pad(xi, ((1, 0), (1, 0)), mode="wrap")
        F_ijb = jax.lax.conv_general_dilated(
            xi[None, :,:, None],
            K,
            (1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWOI", "NHWC"),
        )[0]
        b = -F_ijb[T[:, 0], T[:, 1], T[:, 2]]
        res = solve_step(b)
        return res.at[M].get()

    return solve
