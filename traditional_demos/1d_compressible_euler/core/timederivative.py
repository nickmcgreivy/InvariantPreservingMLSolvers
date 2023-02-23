import jax.numpy as jnp
import jax
from jax import vmap

from flux import Flux
from boundaryconditions import BoundaryCondition
from helper import get_p, get_u, get_H, get_c, get_w, get_entropy_flux
from initialconditions import get_u_left, get_u_right



def ghost_pad(a, n):
	mode = 'edge'
	return jnp.pad(a, ((0,0), (n,n)), mode=mode)

def open_pad(a, n, core_params):
	mode = 'constant'
	aL = get_u_left(core_params)
	aR = get_u_right(core_params)

	rho = a[0]
	rhov = a[1]
	E = a[2]
	rho = jnp.pad(rho, (n, 0), mode=mode, constant_values=(aL[0],))
	rho = jnp.pad(rho, (0, n), mode=mode, constant_values=(aR[0],))
	rhov = jnp.pad(rhov, (n, 0), mode=mode, constant_values=(aL[1],))
	rhov = jnp.pad(rhov, (0, n), mode=mode, constant_values=(aR[1],))
	E = jnp.pad(E, (n, 0), mode=mode, constant_values=(aL[2],))
	E = jnp.pad(E, (0, n), mode=mode, constant_values=(aR[2],))
	return jnp.concatenate([rho[None], rhov[None], E[None]], axis=0)

def closed_pad(a, n):
	mode = 'constant'
	aL = jnp.asarray([a[0,0], -a[1,0], a[2,0]])
	aR = jnp.asarray([a[0,-1], -a[1,-1], a[2,-1]])

	rho = a[0]
	rhov = a[1]
	E = a[2]
	rho = jnp.pad(rho, (n, 0), mode=mode, constant_values=(aL[0],))
	rho = jnp.pad(rho, (0, n), mode=mode, constant_values=(aR[0],))
	rhov = jnp.pad(rhov, (n, 0), mode=mode, constant_values=(aL[1],))
	rhov = jnp.pad(rhov, (0, n), mode=mode, constant_values=(aR[1],))
	E = jnp.pad(E, (n, 0), mode=mode, constant_values=(aL[2],))
	E = jnp.pad(E, (0, n), mode=mode, constant_values=(aR[2],))
	return jnp.concatenate([rho[None], rhov[None], E[None]], axis=0)

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
	a = ghost_pad(a, 1)
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


def flux_musclconserved_nonperiodic(a, core_params):
	da_j_minus = a[:,1:-1] - a[:, :-2]
	da_j_plus  = a[:, 2:]  - a[:, 1:-1]
	a = a[:, 1:-1]
	da_j = vmap_minmod_3(da_j_minus, (da_j_plus + da_j_minus) / 4, da_j_plus)

	da_j = limit_da(a, da_j, core_params)

	aL = (a + da_j)[:,:-1]
	aR = (a - da_j)[:, 1:]
	F  = flux_roe(aL, aR, core_params)
	return F

def flux_musclconserved_ghost(a, core_params):
	a = ghost_pad(a, 2)
	return flux_musclconserved_nonperiodic(a, core_params)

def flux_musclconserved_open(a, core_params):
	a = open_pad(a, 2, core_params)
	return flux_musclconserved_nonperiodic(a, core_params)

def flux_musclconserved_closed(a, core_params):
	a = closed_pad(a, 2)
	return flux_musclconserved_nonperiodic(a, core_params)


def flux_musclconserved_periodic(a, core_params):

	da_j_minus = a - jnp.roll(a, 1, axis=1)
	da_j_plus = jnp.roll(a, -1, axis=1) - a
	da_j = vmap_minmod_3(da_j_minus, (da_j_plus + da_j_minus) / 4, da_j_plus)

	da_j = limit_da(a, da_j, core_params)

	aL = a + da_j
	aR = jnp.roll(a - da_j, -1, axis=1)
	F_R  = flux_roe(aL, aR, core_params)
	return F_R


def flux_musclprimitive_nonperiodic(a, core_params):
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

def flux_musclprimitive_ghost(a, core_params):
	a = ghost_pad(a, 2)
	return flux_musclprimitive_nonperiodic(a, core_params)

def flux_musclprimitive_open(a, core_params):
	a = open_pad(a, 2, core_params)
	return flux_musclprimitive_nonperiodic(a, core_params)

def flux_musclprimitive_closed(a, core_params):
	a = closed_pad(a, 2)
	return flux_musclprimitive_nonperiodic(a, core_params)




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


def flux_musclcharacteristic_nonperiodic(a, core_params):
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


def flux_musclcharacteristic_ghost(a, core_params):
	a = ghost_pad(a, 2)
	return flux_musclcharacteristic_nonperiodic(a, core_params)

def flux_musclcharacteristic_open(a, core_params):
	a = open_pad(a, 2, core_params)
	return flux_musclcharacteristic_nonperiodic(a, core_params)

def flux_musclcharacteristic_closed(a, core_params):
	a = closed_pad(a, 2)
	return flux_musclcharacteristic_nonperiodic(a, core_params)


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


def _time_derivative_euler_periodic(core_params, dt_fn=None):
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
	else:
		raise NotImplementedError
	return flux_term


def _time_derivative_euler_ghost(core_params, dt_fn=None):
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
	else:
		raise NotImplementedError

	return flux_term


def _time_derivative_euler_open(core_params, dt_fn=None):
	if core_params.flux == Flux.MUSCLCONSERVED:
		flux_term = lambda a: flux_musclconserved_open(a, core_params)
	elif core_params.flux == Flux.MUSCLPRIMITIVE:
		flux_term = lambda a: flux_musclprimitive_open(a, core_params)
	elif core_params.flux == Flux.MUSCLCHARACTERISTIC:
		flux_term = lambda a: flux_musclcharacteristic_open(a, core_params)
	else:
		raise NotImplementedError

	return flux_term

def _time_derivative_euler_closed(core_params, dt_fn=None):
	if core_params.flux == Flux.MUSCLCONSERVED:
		flux_term = lambda a: flux_musclconserved_closed(a, core_params)
	elif core_params.flux == Flux.MUSCLPRIMITIVE:
		flux_term = lambda a: flux_musclprimitive_closed(a, core_params)
	elif core_params.flux == Flux.MUSCLCHARACTERISTIC:
		flux_term = lambda a: flux_musclcharacteristic_closed(a, core_params)
	else:
		raise NotImplementedError

	return flux_term


def time_derivative_FV_1D_euler(core_params, dt_fn = None, deta_dt_ratio = None, G = None):

	if core_params.bc == BoundaryCondition.GHOST:
		flux_term = _time_derivative_euler_ghost(core_params, dt_fn = dt_fn)
		def dadt(a):
			nx = a.shape[1]
			dx = core_params.Lx / nx
			F = flux_term(a)# (3, nx + 1)
			if G is not None:
				assert deta_dt_ratio is not None
				G_R = G(a, core_params) # (3, nx-1)
				w = get_w(a, core_params) # (3, nx)
				diff_w = (w[:,1:] - w[:,:-1])
				deta_dt_old = jnp.sum(F[:,1:-1] * diff_w)
				deta_dt_new = deta_dt_ratio * deta_dt_old# + jnp.sum(F[:, -1] * w[:, -1]) - jnp.sum(F[:, 0] * w[:, 0])
				denom = jnp.sum(G_R * diff_w)
				F = F.at[:,1:-1].add((deta_dt_new - deta_dt_old) * G_R / denom)
			F_R = F[:, 1:]
			F_L = F[:, :-1]
			return (F_L - F_R) / dx
	elif core_params.bc == BoundaryCondition.OPEN:
		flux_term = _time_derivative_euler_open(core_params, dt_fn = dt_fn)
		def dadt(a):
			nx = a.shape[1]
			dx = core_params.Lx / nx
			F = flux_term(a)# (3, nx + 1)
			if G is not None:
				assert deta_dt_ratio is not None
				G_R = G(a, core_params) # (3, nx-1)
				w = get_w(a, core_params) # (3, nx)
				diff_w = (w[:,1:] - w[:,:-1])
				deta_dt_old = jnp.sum(F[:,1:-1] * diff_w) - jnp.sum(F[:, -1] * w[:, -1]) + jnp.sum(F[:, 0] * w[:, 0])
				deta_dt_bc = get_entropy_flux(get_u_left(core_params), core_params) - get_entropy_flux(get_u_right(core_params), core_params)
				deta_dt_new =  deta_dt_bc + deta_dt_ratio * (deta_dt_old - deta_dt_bc)
				denom = jnp.sum(G_R * diff_w)
				F = F.at[:,1:-1].add((deta_dt_new - deta_dt_old) * G_R / denom)
			F_R = F[:, 1:]
			F_L = F[:, :-1]
			return (F_L - F_R) / dx
	elif core_params.bc == BoundaryCondition.CLOSED:
		flux_term = _time_derivative_euler_closed(core_params, dt_fn = dt_fn)
		def dadt(a):
			nx = a.shape[1]
			dx = core_params.Lx / nx
			F = flux_term(a)# (3, nx + 1)
			if G is not None:
				assert deta_dt_ratio is not None
				G_R = G(a, core_params) # (3, nx-1)
				w = get_w(a, core_params) # (3, nx)
				diff_w = (w[:,1:] - w[:,:-1])
				deta_dt_old = jnp.sum(F[:,1:-1] * diff_w) - jnp.sum(F[:, -1] * w[:, -1]) + jnp.sum(F[:, 0] * w[:, 0])
				entropy_flux_estimated = 0.0 - 0.0
				deta_dt_new = deta_dt_ratio * (deta_dt_old - entropy_flux_estimated)
				denom = jnp.sum(G_R * diff_w)
				F = F.at[:,1:-1].add((deta_dt_new - deta_dt_old) * G_R / denom)
			F_R = F[:, 1:]
			F_L = F[:, :-1]
			return (F_L - F_R) / dx

	elif core_params.bc == BoundaryCondition.PERIODIC:
		flux_term = _time_derivative_euler_periodic(core_params, dt_fn = dt_fn)
		def dadt(a):
			nx = a.shape[1]
			dx = core_params.Lx / nx
			F_R = jnp.nan_to_num(flux_term(a)) 
			if G is not None:
				assert deta_dt_ratio is not None
				G_R = G(a, core_params) # (3, nx)
				w = get_w(a, core_params) # (3, nx)
				w_plus_one = jnp.roll(w, -1, axis=-1) 
				diff_w = (w_plus_one - w)
				deta_dt_old = jnp.sum(F_R * diff_w)
				deta_dt_new = deta_dt_ratio * deta_dt_old
				denom = jnp.sum(G_R * diff_w)
				F_R = F_R + (deta_dt_new - deta_dt_old) * G_R / denom
			F_L = jnp.roll(F_R, 1, axis=1)
			return (F_L - F_R) / dx
	else:
		raise NotImplementedError

	return dadt