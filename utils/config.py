from pydantic import BaseModel, Field, PrivateAttr

from .constants import ControllerType, LocalizationMode, PathType


class Config(BaseModel):
    localization_mode: LocalizationMode = LocalizationMode.EKF
    controller_type: ControllerType = ControllerType.PID
    path_type: PathType = PathType.SPORADIC
    training_iteration: int = 2
    klp: float = 0.2 
    klv: float = 0.5 
    kli: float = 0.2 
    kap: float = 0.8 
    kav: float = 0.6 
    kai: float = 0.2

    alpha: float = 1e-3 
    kappa: int = 0
    beta: int = 2


    

_config = Config()