from enum import Enum


class BoundaryCondition(Enum):
    PERIODIC = "periodic"
    GHOST = "ghost"
    CLOSED = "closed"
    OPEN = "open"

    def __str__(self):
        return self.value
