from enum import Enum

class Flux(Enum):
    """
    Flux is a subclass of Enum, which determines the flux that is used to compute
    the time-derivative of the equation.
    """

    UPWIND = "upwind"
    CENTERED = "centered"
    MUSCL = "muscl"
    LEARNEDLIMITER = "learnedlimiter"

    def __str__(self):
        return self.value