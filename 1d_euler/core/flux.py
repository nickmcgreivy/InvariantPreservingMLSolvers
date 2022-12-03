from enum import Enum

class Flux(Enum):

    LAXFRIEDRICHS = 'laxfriedrichs'
    ROE = 'roe'

    def __str__(self):
        return self.value