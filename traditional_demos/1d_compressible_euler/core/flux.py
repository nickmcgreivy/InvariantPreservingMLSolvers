from enum import Enum


class Flux(Enum):
    LAXFRIEDRICHS = "laxfriedrichs"
    ROE = "roe"
    RUSANOV = "rusanov"
    MUSCLCONSERVED = "musclconserved"
    MUSCLPRIMITIVE = "musclprimitive"
    MUSCLCHARACTERISTIC = "musclcharacteristic"

    def __str__(self):
        return self.value
