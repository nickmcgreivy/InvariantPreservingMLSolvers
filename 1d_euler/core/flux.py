from enum import Enum

class Flux(Enum):

    LAXFRIEDRICHS = 'laxfriedrichs'
    ROE = 'roe'
    RUSANOV = 'rusanov'
    MUSCL = 'muscl'

    def __str__(self):
        return self.value
