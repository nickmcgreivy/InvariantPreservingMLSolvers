import jax.numpy as jnp
import jax
from jax import vmap

from flux import Flux
from boundaryconditions import BoundaryCondition
from helper import get_p, get_u, get_H, get_c, get_w, has_negative, get_u_from_w, _fixed_quad, get_entropy_flux
from model import model_flux_FV_1D_euler

from jax import config
config.update("jax_enable_x64", True)

def pad_open(a, aL, aR, core_params):
    mode = 'constant'

    rho = a[0]
    rhov = a[1]
    E = a[2]
    rho = jnp.pad(rho, (2, 0), mode=mode, constant_values=(aL[0],))
    rho = jnp.pad(rho, (0, 2), mode=mode, constant_values=(aR[0],))
    rhov = jnp.pad(rhov, (2, 0), mode=mode, constant_values=(aL[1],))
    rhov = jnp.pad(rhov, (0, 2), mode=mode, constant_values=(aR[1],))
    E = jnp.pad(E, (2, 0), mode=mode, constant_values=(aL[2],))
    E = jnp.pad(E, (0, 2), mode=mode, constant_values=(aR[2],))
    return jnp.concatenate([rho[None], rhov[None], E[None]], axis=0)

def pad_ghost(a):
    return jnp.pad(a, ((0,0), (2,2)), mode='edge')


def minmod_3(z1, z2, z3):
    s = (
        0.5
        * (jnp.sign(z1) + jnp.sign(z2))
        * jnp.absolute(0.5 * ((jnp.sign(z1) + jnp.sign(z3))))
    )
    return s * jnp.minimum(jnp.absolute(z1), jnp.minimum(jnp.absolute(z2), jnp.absolute(z3)))

vmap_minmod_3 = vmap(minmod_3, (0, 0, 0), 0)

def f_j(a_j, core_params):
    rho_u = a_j[1]
    rho_usq = rho_u**2 / a_j[0]
    E = a_j[2]
    u = get_u(a_j, core_params)
    p = get_p(a_j, core_params)
    return jnp.asarray([rho_u, rho_usq + p, u * (p + E)])

def flux_laxfriedrichs(aL, aR, core_params, dt, dx):
    return 0.5 * (f_j(aL, core_params) + f_j(aR, core_params)) - 0.5 * (dx / dt) * (aR - aL)

def flux_rusanov(aL, aR, core_params):
    local_max_speed = jnp.maximum(jnp.abs(get_u(aL, core_params)) + get_c(aL, core_params), (jnp.abs(get_u(aR, core_params)) + get_c(aR, core_params)))
    return 0.5 * (f_j(aL, core_params) + f_j(aR, core_params)) - 0.5 * local_max_speed * (aR - aL)

def flux_roe(aL, aR, core_params):

    def entropy_fix(eig, eigL, eigR):
        delta = jnp.maximum(0, jnp.maximum(eig-eigL, eigR - eig))
        abs_eig = jnp.abs(eig)
        return (abs_eig >= delta) * abs_eig + (abs_eig < delta) * delta #jnp.nan_to_num(0.5 * (delta + eig**2 / delta))

    rhoL = aL[0]
    rhoR = aR[0]

    deltap = get_p(aR, core_params) - get_p(aL, core_params)
    deltau = get_u(aR, core_params) - get_u(aL, core_params)
    deltarho = rhoR - rhoL

    rhoRoe = jnp.sqrt(rhoL * rhoR)
    denom = jnp.sqrt(rhoL) + jnp.sqrt(rhoR)
    uRoe = (jnp.sqrt(rhoL) * get_u(aL, core_params) + jnp.sqrt(rhoR) * get_u(aR, core_params)) / denom
    HRoe = (jnp.sqrt(rhoL) * get_H(aL, core_params) + jnp.sqrt(rhoR) * get_H(aR, core_params)) / denom
    cRoe = jnp.sqrt( (core_params.gamma - 1) * (HRoe - uRoe ** 2 / 2) )

    V1 = (deltap - rhoRoe * cRoe * deltau) / (2 * cRoe**2)
    V2 = -(deltap - cRoe**2 * deltarho)     / (cRoe**2)
    V3 = (deltap + rhoRoe * cRoe * deltau) / (2 * cRoe**2)

    ones = jnp.ones(aL.shape[1])
    r1 = jnp.asarray([ones, uRoe - cRoe, HRoe - uRoe * cRoe])
    r2 = jnp.asarray([ones, uRoe,        uRoe**2 / 2       ])
    r3 = jnp.asarray([ones, uRoe + cRoe, HRoe + uRoe * cRoe])

    eig1 = uRoe - cRoe
    eig1L = get_u(aL, core_params) - get_c(aL, core_params)
    eig1R = get_u(aR, core_params) - get_c(aR, core_params)
    eig2 = uRoe
    eig2L = get_u(aL, core_params)
    eig2R = get_u(aR, core_params)
    eig3 = uRoe + cRoe
    eig3L = get_u(aL, core_params) + get_c(aL, core_params)
    eig3R = get_u(aR, core_params) + get_c(aR, core_params)

    corr1 = entropy_fix(eig1, eig1L, eig1R) * V1 * r1
    corr2 = entropy_fix(eig2, eig2L, eig2R) * V2 * r2
    corr3 = entropy_fix(eig3, eig3L, eig3R) * V3 * r3

    return 0.5 * (f_j(aL, core_params) + f_j(aR, core_params)) - 0.5 * (corr1 + corr2 + corr3)


def flux_periodic(a, core_params, flux_fn):
    a_j = a
    a_j_plus_one = jnp.roll(a, -1, axis=1)
    F_R = flux_fn(a_j, a_j_plus_one, core_params)
    return F_R


def flux_ghost(a, core_params, flux_fn):
    a = jnp.pad(a, ((0,0), (1,1)), mode='edge')
    a_j = a[:, :-1]
    a_j_plus_one = a[:, 1:]
    F = flux_fn(a_j, a_j_plus_one, core_params)
    return F

###### 
# MUSCL FLUXES 
######

def limit_da(a, da, core_params):
    ap = a + da
    am = a - da
    pp = get_p(ap, core_params)
    pm = get_p(am, core_params)
    return ~((pp < 0) | (pm < 0) | (ap[0] < 0) | (am[0] < 0)) * da

def limit_dV(V, dV, core_params):
    Vp = V + dV
    Vm = V - dV
    return ~((Vp[0] < 0) | (Vm[0] < 0) | (Vp[2] < 0) | (Vm[2] < 0)) * dV

def flux_musclconserved_ghost(a, core_params):
    a = jnp.pad(a, ((0,0), (2,2)), mode='edge')

    da_j_minus = a[:,1:-1] - a[:, :-2]
    da_j_plus  = a[:, 2:]  - a[:, 1:-1]
    a = a[:, 1:-1]
    da_j = vmap_minmod_3(da_j_minus, (da_j_plus + da_j_minus) / 4, da_j_plus)

    da_j = limit_da(a, da_j, core_params)

    aL = (a + da_j)[:,:-1]
    aR = (a - da_j)[:, 1:]
    F  = flux_roe(aL, aR, core_params)
    return F


def flux_musclconserved_periodic(a, core_params):

    da_j_minus = a - jnp.roll(a, 1, axis=1)
    da_j_plus = jnp.roll(a, -1, axis=1) - a
    da_j = vmap_minmod_3(da_j_minus, (da_j_plus + da_j_minus) / 4, da_j_plus)

    da_j = limit_da(a, da_j, core_params)

    aL = a + da_j
    aR = jnp.roll(a - da_j, -1, axis=1)
    F_R  = flux_roe(aL, aR, core_params)
    return F_R


def flux_musclprimitive_ghost(a, core_params):
    a = jnp.pad(a, ((0,0), (2,2)), mode='edge')

    rho = a[0]
    u = get_u(a, core_params)
    p = get_p(a, core_params)
    V = jnp.asarray([rho, u, p])

    dV_j_minus = V[:,1:-1] - V[:, :-2]
    dV_j_plus  = V[:, 2:]  - V[:, 1:-1]
    V = V[:, 1:-1]
    dV_j = vmap_minmod_3(dV_j_minus, (dV_j_plus + dV_j_minus) / 4, dV_j_plus)

    dV_j = limit_dV(V, dV_j, core_params)

    VL = (V + dV_j)[:,:-1]
    VR = (V - dV_j)[:, 1:]

    EL = VL[2]/(core_params.gamma - 1) + 0.5 * VL[0] * VL[1]**2
    ER = VR[2]/(core_params.gamma - 1) + 0.5 * VR[0] * VR[1]**2
    aL = jnp.asarray([VL[0], VL[0] * VL[1], EL])
    aR = jnp.asarray([VR[0], VR[0] * VR[1], ER])

    F  = flux_roe(aL, aR, core_params)
    return F


def flux_musclprimitive_periodic(a, core_params):
    rho = a[0]
    u = get_u(a, core_params)
    p = get_p(a, core_params)
    V = jnp.asarray([rho, u, p])

    dV_j_minus = V - jnp.roll(V, 1, axis=1)
    dV_j_plus = jnp.roll(V, -1, axis=1) - V
    dV_j = vmap_minmod_3(dV_j_minus, (dV_j_plus + dV_j_minus) / 4, dV_j_plus)

    dV_j = limit_dV(V, dV_j, core_params)

    VL = V + dV_j
    VR = jnp.roll(V - dV_j, -1, axis=1)

    EL = VL[2]/(core_params.gamma - 1) + 0.5 * VL[0] * VL[1]**2
    ER = VR[2]/(core_params.gamma - 1) + 0.5 * VR[0] * VR[1]**2
    aL = jnp.asarray([VL[0], VL[0] * VL[1], EL])
    aR = jnp.asarray([VR[0], VR[0] * VR[1], ER])

    F_R = flux_roe(aL, aR, core_params)
    return F_R


def flux_musclcharacteristic_ghost(a, core_params):
    a = pad_ghost(a)

    dQ_minus = a[:,1:-1] - a[:, :-2]
    dQ_plus = a[:, 2:] - a[:, 1:-1]

    a = a[:,1:-1]

    h = get_H(a, core_params)
    u = get_u(a, core_params)
    c = get_c(a, core_params)
    b = core_params.gamma - 1

    #### See SimJournal 4 for notation http://ammar-hakim.org/sj/euler-eigensystem.html

    alpha_1_minus = (b / c**2) * ( (h-u**2) * dQ_minus[0] + u * dQ_minus[1] - dQ_minus[2] )
    alpha_2_minus = 1 / (2 * c) * (dQ_minus[1] +(c - u) * dQ_minus[0] - c * alpha_1_minus )
    alpha_0_minus = dQ_minus[0] - alpha_1_minus - alpha_2_minus
    dD_minus = jnp.asarray([alpha_0_minus, alpha_1_minus, alpha_2_minus]) # D stands for Delta = L(Q_i) (Q_{...}-Q_{...})
    
    alpha_1_plus = (b / c**2) * ( (h-u**2) * dQ_plus[0] + u * dQ_plus[1] - dQ_plus[2] )
    alpha_2_plus = 1 / (2 * c) * (dQ_plus[1] + (c - u) * dQ_plus[0] - c * alpha_1_plus )
    alpha_0_plus = dQ_plus[0] - alpha_1_plus - alpha_2_plus
    dD_plus = jnp.asarray([alpha_0_plus, alpha_1_plus, alpha_2_plus])

    dD = vmap_minmod_3(dD_minus, (dD_plus + dD_minus) / 4, dD_plus)

    ones = jnp.ones(a.shape[1])
    r1 = jnp.asarray([ones, u - c, h - u * c])
    r2 = jnp.asarray([ones, u,     u**2 / 2 ])
    r3 = jnp.asarray([ones, u + c, h + u * c])
    R = jnp.asarray([r1, r2, r3])

    da_j = jnp.einsum('ijk,ik->jk', R, dD)

    da_j = limit_da(a, da_j, core_params)

    aL = (a + da_j)[:,:-1]
    aR = (a - da_j)[:, 1:]
    F  = flux_roe(aL, aR, core_params)
    return F

def flux_musclcharacteristic_open(a, aL, aR, core_params):
    a = pad_open(a, aL, aR, core_params)


    dQ_minus = a[:,1:-1] - a[:, :-2]
    dQ_plus = a[:, 2:] - a[:, 1:-1]

    a = a[:,1:-1]

    h = get_H(a, core_params)
    u = get_u(a, core_params)
    c = get_c(a, core_params)
    b = core_params.gamma - 1

    #### See SimJournal 4 for notation http://ammar-hakim.org/sj/euler-eigensystem.html

    alpha_1_minus = (b / c**2) * ( (h-u**2) * dQ_minus[0] + u * dQ_minus[1] - dQ_minus[2] )
    alpha_2_minus = 1 / (2 * c) * (dQ_minus[1] +(c - u) * dQ_minus[0] - c * alpha_1_minus )
    alpha_0_minus = dQ_minus[0] - alpha_1_minus - alpha_2_minus
    dD_minus = jnp.asarray([alpha_0_minus, alpha_1_minus, alpha_2_minus]) # D stands for Delta = L(Q_i) (Q_{...}-Q_{...})
    
    alpha_1_plus = (b / c**2) * ( (h-u**2) * dQ_plus[0] + u * dQ_plus[1] - dQ_plus[2] )
    alpha_2_plus = 1 / (2 * c) * (dQ_plus[1] + (c - u) * dQ_plus[0] - c * alpha_1_plus )
    alpha_0_plus = dQ_plus[0] - alpha_1_plus - alpha_2_plus
    dD_plus = jnp.asarray([alpha_0_plus, alpha_1_plus, alpha_2_plus])

    dD = vmap_minmod_3(dD_minus, (dD_plus + dD_minus) / 4, dD_plus)

    ones = jnp.ones(a.shape[1])
    r1 = jnp.asarray([ones, u - c, h - u * c])
    r2 = jnp.asarray([ones, u,     u**2 / 2 ])
    r3 = jnp.asarray([ones, u + c, h + u * c])
    R = jnp.asarray([r1, r2, r3])

    da_j = jnp.einsum('ijk,ik->jk', R, dD)

    da_j = limit_da(a, da_j, core_params)

    aL = (a + da_j)[:,:-1]
    aR = (a - da_j)[:, 1:]
    F  = flux_roe(aL, aR, core_params)
    return F


def flux_musclcharacteristic_periodic(a, core_params):

    dQ_minus = a - jnp.roll(a, 1, axis=1)
    dQ_plus = jnp.roll(a, -1, axis=1) - a
    
    h = get_H(a, core_params)
    u = get_u(a, core_params)
    c = get_c(a, core_params)
    b = core_params.gamma - 1

    alpha_1_minus = (b / c**2) * ( (h-u**2) * dQ_minus[0] + u * dQ_minus[1] - dQ_minus[2] )
    alpha_2_minus = 1 / (2 * c) * (dQ_minus[1] +(c - u) * dQ_minus[0] - c * alpha_1_minus )
    alpha_0_minus = dQ_minus[0] - alpha_1_minus - alpha_2_minus
    dD_minus = jnp.asarray([alpha_0_minus, alpha_1_minus, alpha_2_minus]) # D stands for Delta = L(Q_i) (Q_{...}-Q_{...})
    
    alpha_1_plus = (b / c**2) * ( (h-u**2) * dQ_plus[0] + u * dQ_plus[1] - dQ_plus[2] )
    alpha_2_plus = 1 / (2 * c) * (dQ_plus[1] + (c - u) * dQ_plus[0] - c * alpha_1_plus )
    alpha_0_plus = dQ_plus[0] - alpha_1_plus - alpha_2_plus
    dD_plus = jnp.asarray([alpha_0_plus, alpha_1_plus, alpha_2_plus])

    dD = vmap_minmod_3(dD_minus, (dD_plus + dD_minus) / 4, dD_plus)

    ones = jnp.ones(a.shape[1])
    r1 = jnp.asarray([ones, u - c, h - u * c])
    r2 = jnp.asarray([ones, u,     u**2 / 2 ])
    r3 = jnp.asarray([ones, u + c, h + u * c])
    R = jnp.asarray([r1, r2, r3])

    da_j = jnp.einsum('ijk,ik->jk', R, dD)

    da_j = limit_da(a, da_j, core_params)

    aL = a + da_j
    aR = jnp.roll(a - da_j, -1, axis=1)
    F_R  = flux_roe(aL, aR, core_params)
    return F_R


def flux_ep_ghost(a, core_params):
    a = jnp.pad(a, ((0,0), (1,1)), mode='edge')
    flux_ep = entropy_preserving_flux_ghost(a, core_params)
    flux_ep = flux_ep.at[:,0].set(f_j(a[:,0], core_params))
    flux_ep = flux_ep.at[:,-1].set(f_j(a[:,-1], core_params))
    return flux_ep


def flux_learned_periodic(a, core_params, model = None, params = None):
    return model_flux_FV_1D_euler(a, model, params)

def flux_learned_nonperiodic(a, core_params, model=None, params=None):
    return model_flux_FV_1D_euler(a, model, params)


def _time_derivative_euler_periodic(core_params, model=None, params=None, dt_fn=None):
    if core_params.flux == Flux.MUSCLCONSERVED:
        flux_term = lambda a: flux_musclconserved_periodic(a, core_params)
    elif core_params.flux == Flux.MUSCLPRIMITIVE:
        flux_term = lambda a: flux_musclprimitive_periodic(a, core_params)
    elif core_params.flux == Flux.MUSCLCHARACTERISTIC:
        flux_term = lambda a: flux_musclcharacteristic_periodic(a, core_params)
    elif core_params.flux == Flux.LAXFRIEDRICHS:
        assert dt_fn is not None
        flux_fn = lambda aL, aR, core_params: flux_laxfriedrichs(aL, aR, core_params, dt_fn(aL), core_params.Lx / aL.shape[1])
        flux_term = lambda a: flux_periodic(a, core_params, flux_fn)
    elif core_params.flux == Flux.ROE:
        flux_fn = flux_roe
        flux_term = lambda a: flux_periodic(a, core_params, flux_fn)
    elif core_params.flux == Flux.RUSANOV:
        flux_fn = flux_rusanov
        flux_term = lambda a: flux_periodic(a, core_params, flux_fn)
    elif core_params.flux == Flux.LEARNED:
        def flux_term(a):
            flux_right = flux_musclcharacteristic_periodic(a, core_params)
            delta_flux_right = flux_learned_periodic(a, core_params, model = model, params = params)
            return flux_right + delta_flux_right
    else:
        raise NotImplementedError
    return flux_term


def _time_derivative_euler_ghost(core_params, model=None, params=None, dt_fn=None):
    if core_params.flux == Flux.MUSCLCONSERVED:
        flux_term = lambda a: flux_musclconserved_ghost(a, core_params)
    elif core_params.flux == Flux.MUSCLPRIMITIVE:
        flux_term = lambda a: flux_musclprimitive_ghost(a, core_params)
    elif core_params.flux == Flux.MUSCLCHARACTERISTIC:
        flux_term = lambda a: flux_musclcharacteristic_ghost(a, core_params)
    elif core_params.flux == Flux.LAXFRIEDRICHS:
        assert dt_fn is not None
        flux_fn = lambda aL, aR, core_params: flux_laxfriedrichs(aL, aR, core_params, dt_fn(aL), core_params.Lx / aL.shape[1])
        flux_term = lambda a: flux_ghost(a, core_params, flux_fn)
    elif core_params.flux == Flux.ROE:
        flux_fn = flux_roe
        flux_term = lambda a: flux_ghost(a, core_params, flux_fn)
    elif core_params.flux == Flux.RUSANOV:
        flux_fn = flux_rusanov
        flux_term = lambda a: flux_ghost(a, core_params, flux_fn)
    elif core_params.flux == Flux.EP:
        flux_term = lambda a: flux_ep_ghost(a, core_params)
    elif core_params.flux == Flux.LEARNED:
        def flux_term(a):
            flux_right = flux_musclcharacteristic_ghost(a, core_params)
            delta_flux_right = flux_learned_nonperiodic(a, core_params, model = model, params = params)
            return flux_right + delta_flux_right
    else:
        raise NotImplementedError

    return flux_term

def _time_derivative_euler_open(core_params, model=None, params=None, dt_fn=None):
    if core_params.flux == Flux.MUSCLCHARACTERISTIC:
        flux_term = lambda a, aL, aR: flux_musclcharacteristic_open(a, aL, aR, core_params)
    elif core_params.flux == Flux.LEARNED:
        def flux_term(a, aL, aR):
            flux_right = flux_musclcharacteristic_open(a, aL, aR, core_params)
            delta_flux_right = flux_learned_nonperiodic(a, core_params, model = model, params = params)
            return flux_right + delta_flux_right
    else:
        raise NotImplementedError

    return flux_term



def positivity_limiter_periodic(a, flux_right, core_params, dt_fn):
    """
    Limiter from https://www.sciencedirect.com/science/article/pii/S0021999113000557
    """

    def get_a_plus_minus(a, flux_right, core_params, delta):
        return a - 2 * delta * flux_right, jnp.roll(a + 2 * delta * jnp.roll(flux_right, 1, axis=-1), -1, axis=-1)


    def solve_theta(var_LF, var, epsilon):
        init_theta = (var_LF - epsilon) / (var_LF - var)
        valid = (init_theta > 0.0) * (init_theta < 1.0)
        return valid * jnp.nan_to_num(init_theta, nan=0.0, posinf=0.0, neginf=0.0)

    flux_fn = lambda aL, aR, core_params: flux_laxfriedrichs(aL, aR, core_params, dt_fn(a), core_params.Lx / a.shape[1])
    flux_right_LF = flux_periodic(a, core_params, flux_fn)


    nx = a.shape[-1]
    dt = dt_fn(a)
    dx = core_params.Lx / nx
    delta = dt/dx
    init_ones = jnp.ones((nx))

    a_plus, a_minus = get_a_plus_minus(a, flux_right, core_params, delta)
    rho_plus = a_plus[0]
    rho_minus = a_minus[0]


    a_LF_plus, a_LF_minus = get_a_plus_minus(a, flux_right_LF, core_params, delta)
    rho_LF_plus = a_LF_plus[0]
    rho_LF_minus = a_LF_minus[0]
    p_LF_plus = get_p(a_LF_plus, core_params)
    p_LF_minus = get_p(a_LF_minus, core_params)

    ### ensure positive density
    epsilon_rho = 1e-3

    below_zero = rho_plus < epsilon_rho
    above_zero = rho_plus >= epsilon_rho
    theta_plus_below = solve_theta(rho_LF_plus, rho_plus, epsilon_rho)
    theta_plus = below_zero * theta_plus_below + above_zero * init_ones

    

    below_zero = rho_minus < epsilon_rho
    above_zero = rho_minus >= epsilon_rho
    theta_minus_below = solve_theta(rho_LF_minus, rho_minus, epsilon_rho)
    theta_minus = below_zero * theta_minus_below + above_zero * init_ones

    theta_rho = jnp.minimum(theta_plus, theta_minus)


    flux_right_star = theta_rho * jnp.nan_to_num(flux_right) + (init_ones - theta_rho) * flux_right_LF
    a_star_plus, a_star_minus = get_a_plus_minus(a, flux_right_star, core_params, delta)
    p_star_plus = get_p(a_star_plus, core_params)
    p_star_minus = get_p(a_star_minus, core_params)


    ### ensure positive pressure
    epsilon_p = 1e-3

    below_zero = p_star_plus < epsilon_p
    above_zero = p_star_plus >= epsilon_p
    theta_plus_below = solve_theta(p_LF_plus, p_star_plus, epsilon_p)
    theta_plus = below_zero * theta_plus_below + above_zero * init_ones


    below_zero = p_star_minus < epsilon_p
    above_zero = p_star_minus >= epsilon_p
    theta_minus_below = solve_theta(p_LF_minus, p_star_minus, epsilon_p)
    theta_minus = below_zero * theta_minus_below + above_zero * init_ones

    theta_p = jnp.minimum(theta_plus, theta_minus)

    flux_return = theta_p * flux_right_star + (init_ones - theta_p) * flux_right_LF
    return flux_return


def positivity_limiter_nonperiodic(a, flux_right, core_params, dt_fn):

    def get_a_plus_minus(a, flux_right, core_params, delta):
        # a is (3, nx+1), F is (3, nx+1), want to return 
        return (a - 2 * delta * flux_right[:,1:])[:,:-1], (a + 2 * delta * flux_right[:,:-1])[:,1:]

    def solve_theta(var_LF, var, epsilon):
        init_theta = (var_LF - epsilon) / (var_LF - var)
        valid = (init_theta > 0.0) * (init_theta < 1.0)
        return valid * jnp.nan_to_num(init_theta, nan=0.0, posinf=0.0, neginf=0.0)

    flux_fn = lambda aL, aR, core_params: flux_laxfriedrichs(aL, aR, core_params, dt_fn(a), core_params.Lx / a.shape[1])
    flux_right_LF = flux_ghost(a, core_params, flux_fn) # (3, nx + 1)

    nx = a.shape[-1] # (3, nx)
    dt = dt_fn(a)
    dx = core_params.Lx / nx
    delta = dt/dx
    init_ones = jnp.ones((nx - 1))

    a_plus, a_minus = get_a_plus_minus(a, flux_right, core_params, delta) #(3, nx-1)
    rho_plus = a_plus[0]
    rho_minus = a_minus[0]


    a_LF_plus, a_LF_minus = get_a_plus_minus(a, flux_right_LF, core_params, delta) #(3, nx-1)
    rho_LF_plus = a_LF_plus[0]
    rho_LF_minus = a_LF_minus[0]
    p_LF_plus = get_p(a_LF_plus, core_params)
    p_LF_minus = get_p(a_LF_minus, core_params)

    ### ensure positive density
    epsilon_rho = 1e-3

    below_zero = rho_plus < epsilon_rho 
    above_zero = rho_plus >= epsilon_rho
    theta_plus_below = solve_theta(rho_LF_plus, rho_plus, epsilon_rho)
    theta_plus = jnp.ones((nx + 1))
    theta_plus = theta_plus.at[1:-1].set(below_zero * theta_plus_below + above_zero * init_ones)

    below_zero = rho_minus < epsilon_rho
    above_zero = rho_minus >= epsilon_rho
    theta_minus_below =  solve_theta(rho_LF_minus, rho_minus, epsilon_rho)
    theta_minus = jnp.ones((nx + 1))
    theta_minus = theta_minus.at[1:-1].set(below_zero * theta_minus_below + above_zero * init_ones)

    theta_rho = jnp.minimum(theta_plus, theta_minus) # (3, nx + 1)

    flux_right_star = theta_rho * jnp.nan_to_num(flux_right) + (1 - theta_rho) * flux_right_LF # (3, nx + 1)

    a_star_plus, a_star_minus = get_a_plus_minus(a, flux_right_star, core_params, delta)
    p_star_plus = get_p(a_star_plus, core_params)
    p_star_minus = get_p(a_star_minus, core_params)

    ### ensure positive pressure
    epsilon_p = 1e-3

    below_zero = p_star_plus < epsilon_p
    above_zero = p_star_plus >= epsilon_p
    theta_plus_below = solve_theta(p_LF_plus, p_star_plus, epsilon_p)
    theta_plus = jnp.ones((nx + 1))
    theta_plus = theta_plus.at[1:-1].set(below_zero * theta_plus_below + above_zero * init_ones)


    below_zero = p_star_minus < epsilon_p
    above_zero = p_star_minus >= epsilon_p
    theta_minus_below = solve_theta(p_LF_minus, p_star_minus, epsilon_p)
    theta_minus = jnp.ones((nx + 1))
    theta_minus = theta_minus.at[1:-1].set(below_zero * theta_minus_below + above_zero * init_ones)

    theta_p = jnp.minimum(theta_plus, theta_minus)

    return theta_p * flux_right_star + (1 - theta_p) * flux_right_LF


def entropy_increase_periodic(a, flux_right, core_params):

    def G_primitive_periodic(a, core_params):
        p = get_p(a, core_params)
        rho = a[0]
        zeros = jnp.zeros(rho.shape)
        u = a[1] / a[0]
        G = jnp.concatenate([zeros[None], u[None], p[None]],axis=0)
        return jnp.roll(G, -1, axis=-1) - G

    G_R = G_primitive_periodic(a, core_params) # (3, nx)
    w = get_w(a, core_params) # (3, nx)
    w_plus_one = jnp.roll(w, -1, axis=-1) 
    diff_w = (w_plus_one - w)
    deta_dt_old = jnp.sum(flux_right * diff_w)
    denom = jnp.sum(G_R * diff_w)
    return flux_right - jnp.nan_to_num((deta_dt_old < 0.0) * deta_dt_old * G_R / denom)



def entropy_preserving_flux_ghost(a, core_params):

    w = get_w(a, core_params)

    nx = a.shape[1]

    w_j = w[:, :-1]
    w_j_plus_one = w[:, 1:]

    def w_hat(w_j, w_j_plus_one, theta):
        return w_j + theta * (w_j_plus_one - w_j)

    def flux_w(w):
        a = get_u_from_w(w, core_params)
        return f_j(a, core_params)

    def f_to_integrate(w_j, w_j_plus_one, theta):
        return flux_w(w_hat(w_j, w_j_plus_one, theta))
    
    
    def integrate(w_j, w_j_plus_one):
        
        def foo(theta):
            return f_to_integrate(w_j, w_j_plus_one, theta)

        return _fixed_quad(vmap(foo, (0), (1)), 0.0, 1.0, n=8)
    
    return vmap(integrate, (1, 1), (1))(w_j, w_j_plus_one)
    



def entropy_increase_ghost(a, flux_right, core_params):

    def G_primitive_ghost(a, core_params):
            p = get_p(a, core_params)
            rho = a[0]
            zeros = jnp.zeros(rho.shape)
            u = a[1] / a[0]
            G = jnp.concatenate([zeros[None], u[None], p[None]], axis=0)
            return G[:,1:] - G[:,:-1]

    G_R = G_primitive_ghost(a, core_params) # (3, nx-1)
    w = get_w(a, core_params) # (3, nx)
    diff_w = (w[:,1:] - w[:,:-1])
    deta_dt_old = jnp.sum(flux_right[:,1:-1] * diff_w) + jnp.sum(flux_right[:,0] * w[:,0]) - jnp.sum(flux_right[:,-1] * w[:,-1])
    deta_dt_new = get_entropy_flux(a[:, 0], core_params) - get_entropy_flux(a[:, -1], core_params)
    denom = jnp.sum(G_R * diff_w)
    flux_right = flux_right.at[:,1:-1].add(jnp.nan_to_num((deta_dt_old < deta_dt_new) * (deta_dt_new - deta_dt_old) * G_R / denom))
    return flux_right


def entropy_increase_open(a, flux_right, core_params, aL, aR):

    def G_primitive_open(a, core_params):
        p = get_p(a, core_params)
        rho = a[0]
        zeros = jnp.zeros(rho.shape)
        u = a[1] / a[0]
        G = jnp.concatenate([zeros[None], u[None], p[None]], axis=0)
        return G[:,1:] - G[:,:-1]

    G_R = G_primitive_open(a, core_params)
    w = get_w(a, core_params) # (3, nx)
    diff_w = (w[:,1:] - w[:,:-1])
    deta_dt_old = jnp.sum(flux_right[:,1:-1] * diff_w) #+ jnp.sum(flux_right[:,0] * w[:,0]) - jnp.sum(flux_right[:,-1] * w[:,-1])
    deta_dt_new = jnp.minimum(get_entropy_flux(aL, core_params), get_entropy_flux(a[:,0], core_params)) - jnp.maximum(get_entropy_flux(aR, core_params), get_entropy_flux(a[:, -1], core_params))
    denom = jnp.sum(G_R * diff_w)
    flux_right = flux_right.at[:,1:-1].add(jnp.nan_to_num((deta_dt_old < deta_dt_new) * (deta_dt_new - deta_dt_old) * G_R / denom))
    return flux_right


def time_derivative_FV_1D_euler(core_params, model=None, params=None, dt_fn = None, invariant_preserving=False):

    if core_params.bc == BoundaryCondition.GHOST:
        flux_term = _time_derivative_euler_ghost(core_params, model=model, params=params, dt_fn = dt_fn)
        def dadt(a, aL=None, aR=None):
            nx = a.shape[1]
            dx = core_params.Lx / nx
            F = flux_term(a) 

            if invariant_preserving == True:
                assert dt_fn is not None
                F = positivity_limiter_nonperiodic(a, F, core_params, dt_fn)
                F = entropy_increase_ghost(a, F, core_params)

            F_R = F[:, 1:]
            F_L = F[:, :-1]
            return (F_L - F_R) / dx

    elif core_params.bc == BoundaryCondition.OPEN:
        flux_term = _time_derivative_euler_open(core_params, model=model, params=params, dt_fn = dt_fn)
        def dadt(a, aL, aR):
            nx = a.shape[1]
            dx = core_params.Lx / nx
            F = flux_term(a, aL, aR)# (3, nx + 1)

            if invariant_preserving == True:
                assert dt_fn is not None
                F = positivity_limiter_nonperiodic(a, F, core_params, dt_fn)
                F = entropy_increase_open(a, F, core_params, aL, aR)

            F_R = F[:, 1:]
            F_L = F[:, :-1]
            return (F_L - F_R) / dx

    elif core_params.bc == BoundaryCondition.PERIODIC:
        flux_term = _time_derivative_euler_periodic(core_params, model=model, params=params, dt_fn = dt_fn)
        def dadt(a, aL=None, aR=None):
            nx = a.shape[1]
            dx = core_params.Lx / nx
            flux_right = flux_term(a) 

            if invariant_preserving == True:
                assert dt_fn is not None
                flux_right = positivity_limiter_periodic(a, flux_right, core_params, dt_fn)
                flux_right = entropy_increase_periodic(a, flux_right, core_params)

            flux_left = jnp.roll(flux_right, 1, axis=1)
            return (flux_left - flux_right) / dx
    else:
        raise NotImplementedError


    return dadt
