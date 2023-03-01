from enum import Enum


class Flux(Enum):
    LAXFRIEDRICHS = "laxfriedrichs"
    ROE = "roe"
    RUSANOV = "rusanov"
    MUSCLCONSERVED = "musclconserved"
    MUSCLPRIMITIVE = "musclprimitive"
    MUSCLCHARACTERISTIC = "musclcharacteristic"
    EP = "ep"
    LEARNED = "learned"

    def __str__(self):
        return self.value
