from enum import Enum


class BoundaryCondition(Enum):
    GHOST = "ghost"
    PERIODIC = "periodic"
    OPEN = "open"
    CLOSED = "closed"

    def __str__(self):
        return self.value
