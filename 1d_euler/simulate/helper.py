import jax.numpy as jnp
from jax import vmap, jit

def _fixed_quad(f, a, b, n=5):
	"""
	Single quadrature of a given order.

	Inputs
	f: function which takes a vector of positions of length n
	and returns a multivariate output of length (3, n)
	a: beginning of integration
	b: end of integration
	n: order of quadrature. max n is 8.
	"""
	assert isinstance(n, int) and n <= 8 and n > 0
	w = {
		1: jnp.asarray([2.0]),
		2: jnp.asarray([1.0, 1.0]),
		3: jnp.asarray(
			[
				0.5555555555555555555556,
				0.8888888888888888888889,
				0.555555555555555555556,
			]
		),
		4: jnp.asarray(
			[
				0.3478548451374538573731,
				0.6521451548625461426269,
				0.6521451548625461426269,
				0.3478548451374538573731,
			]
		),
		5: jnp.asarray(
			[
				0.2369268850561890875143,
				0.4786286704993664680413,
				0.5688888888888888888889,
				0.4786286704993664680413,
				0.2369268850561890875143,
			]
		),
		6: jnp.asarray(
			[
				0.1713244923791703450403,
				0.3607615730481386075698,
				0.4679139345726910473899,
				0.4679139345726910473899,
				0.3607615730481386075698,
				0.1713244923791703450403,
			]
		),
		7: jnp.asarray(
			[
				0.1294849661688696932706,
				0.2797053914892766679015,
				0.38183005050511894495,
				0.417959183673469387755,
				0.38183005050511894495,
				0.279705391489276667901,
				0.129484966168869693271,
			]
		),
		8: jnp.asarray(
			[
				0.1012285362903762591525,
				0.2223810344533744705444,
				0.313706645877887287338,
				0.3626837833783619829652,
				0.3626837833783619829652,
				0.313706645877887287338,
				0.222381034453374470544,
				0.1012285362903762591525,
			]
		),
	}[n]

	xi_i = {
		1: jnp.asarray([0.0]),
		2: jnp.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
		3: jnp.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
		4: jnp.asarray(
			[
				-0.861136311594052575224,
				-0.3399810435848562648027,
				0.3399810435848562648027,
				0.861136311594052575224,
			]
		),
		5: jnp.asarray(
			[
				-0.9061798459386639927976,
				-0.5384693101056830910363,
				0.0,
				0.5384693101056830910363,
				0.9061798459386639927976,
			]
		),
		6: jnp.asarray(
			[
				-0.9324695142031520278123,
				-0.661209386466264513661,
				-0.2386191860831969086305,
				0.238619186083196908631,
				0.661209386466264513661,
				0.9324695142031520278123,
			]
		),
		7: jnp.asarray(
			[
				-0.9491079123427585245262,
				-0.7415311855993944398639,
				-0.4058451513773971669066,
				0.0,
				0.4058451513773971669066,
				0.7415311855993944398639,
				0.9491079123427585245262,
			]
		),
		8: jnp.asarray(
			[
				-0.9602898564975362316836,
				-0.7966664774136267395916,
				-0.5255324099163289858177,
				-0.1834346424956498049395,
				0.1834346424956498049395,
				0.5255324099163289858177,
				0.7966664774136267395916,
				0.9602898564975362316836,
			]
		),
	}[n]

	x_i = (b + a) / 2 + (b - a) / 2 * xi_i
	wprime = w * (b - a) / 2
	return jnp.sum(wprime[None, :] * f(x_i), axis=1)


def map_f_to_FV(f, nx, dx, quad_func=_fixed_quad, n=5, t = 0.0):
	"""
	Takes a function f of type lambda x, t: f(x,t) and
	generates the FV representation of the solution, an
	array of size (3, nx).

	"""
	j = jnp.arange(nx)
	a = dx * j
	b = dx * (j + 1)
	f_vmap = vmap(lambda x: f(x, t), 0, 1)
	_vmap_fixed_quad = vmap(
		lambda f, a, b: quad_func(f_vmap, a, b, n=n), (None, 0, 0), 1
	)  # is n = p+1 high enough order?

	return _vmap_fixed_quad(f_vmap, a, b) / dx


def get_conserved_from_primitive(V, core_params):
	# V is [rho, u, p]:
	rho_u = V[0] * V[1]
	E = V[2] / (core_params.gamma - 1) + 0.5 * V[0] * V[1]**2
	return jnp.asarray([V[0], rho_u, E])

def get_rho(a, core_params):
	return a[0]

def get_u(a, core_params):
	return a[1] / a[0]

def get_p(a, core_params):
	E = a[3]
	p = (E - (1/2 * a[1]**2 / a[0]) ) * (core_params.gamma - 1)
	return p

def get_H(a, core_params):
	return (a[2] + get_p(a, core_params)) / a[0]

def get_c(a, core_params):
	return jnp.sqrt(core_params.gamma * get_p(a, core_params) / get_rho(a, core_params))

def get_entropy(a, core_params):
    p = get_p(a, core_params)
    rho = a[0]
    gamma = core_params.gamma
    return rho * (p / rho**gamma)**(1/(1+gamma))
