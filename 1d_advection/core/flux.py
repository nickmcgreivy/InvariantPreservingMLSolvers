from enum import Enum


class Flux(Enum):
    """
    Flux is a subclass of Enum, which determines the flux that is used to compute
    the time-derivative of the equation.
    """

    UPWIND = "upwind"
    CENTERED = "centered"
    MUSCL = "muscl"
    LEARNED = "learned"
    LEARNEDLIMITER = "learnedlimiter"
    COMBINATION_LEARNED = "combination_learned"

    def __str__(self):
        return self.value
