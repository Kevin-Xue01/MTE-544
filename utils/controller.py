
import numpy as np

from .config import _config
from .helper import (
    calculate_angular_error,
    calculate_linear_error,
)
from .pid import PID


class Controller:
    def __init__(self):
        self.PID_linear = PID(kp=_config.klp, kv=_config.klv, ki=_config.kli)
        self.PID_angular = PID(kp=_config.kap, kv=_config.kav, ki=_config.kai)
    
    def vel_request(self, pose, listGoals):
        goal = self.lookFarFor(pose, listGoals)
        
        finalGoal = listGoals[-1]
        
        e_lin = calculate_linear_error(pose, finalGoal)
        e_ang = calculate_angular_error(pose, goal)
        
        linear_vel = self.PID_linear.update([e_lin, pose[3]], True)
        angular_vel = self.PID_angular.update([e_ang, pose[3]], True) 

        linear_vel = 0.5 if linear_vel > 1.0 else linear_vel
        angular_vel = 0.5 if angular_vel > 1.0 else angular_vel

        return linear_vel, angular_vel


    def lookFarFor(self, pose, listGoals):
        poseArray = np.array([pose[0], pose[1]]) 
        listGoalsArray = np.array(listGoals)

        distanceSquared = np.sum((listGoalsArray-poseArray)**2, axis = 1)
        closestIndex = np.argmin(distanceSquared)

        return listGoals[ min(closestIndex + 3, len(listGoals) - 1) ]
