from enum import Enum
import jax.numpy as np


class Flux(Enum):
    UPWIND = "upwind"
    CENTERED = "centered"
    VANLEER = "vanleer"
    CONSERVATION = "conservation"
    CONSERVATION2 = "conservation2"
    ENERGYCONSERVATION = "energyconservation"
    ENERGYCONSERVATION2 = "energyconservation2"

    def __str__(self):
        return self.value


def minmod(r):
    return np.maximum(0, np.minimum(1, r))


def minmod_2(z1, z2):
    s = 0.5 * (np.sign(z1) + np.sign(z2))
    return s * np.minimum(np.absolute(z1), np.absolute(z2))


def minmod_3(z1, z2, z3):
    s = (
        0.5
        * (np.sign(z1) + np.sign(z2))
        * np.absolute(0.5 * ((np.sign(z1) + np.sign(z3))))
    )
    return s * np.minimum(np.absolute(z1), np.minimum(np.absolute(z2), np.absolute(z3)))
