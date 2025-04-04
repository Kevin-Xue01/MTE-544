from enum import Enum, auto


class LocalizationMode(Enum):
    RAW = auto()
    EKF = auto()
    UKF = auto() 

class ControllerType(Enum):
    P = auto() # poportional
    PD = auto() # proportional and derivative
    PI = auto() # proportional and integral
    PID = auto() # proportional, integral, derivative

class ControllerErrorType(Enum):
    LINEAR = auto()
    ANGULAR = auto()

class PathType(Enum):
    CIRCLE = auto()
    ZIGZAG = auto()
    SPORADIC = auto()
    SQUARE = auto()
    SNAKE = auto()