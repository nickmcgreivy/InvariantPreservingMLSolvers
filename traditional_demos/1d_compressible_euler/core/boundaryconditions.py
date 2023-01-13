from enum import Enum

class BoundaryCondition(Enum):

    GHOST = 'ghost'
    PERIODIC = 'periodic'

    def __str__(self):
        return self.value