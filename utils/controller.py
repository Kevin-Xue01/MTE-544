
import numpy as np

from .helper import (
    LocalizationMode,
    calculate_angular_error,
    calculate_linear_error,
    euler_from_quaternion,
)
from .pid import PID_ctrl

M_PI=3.1415926535

P=0; PD=1; PI=2; PID=3

class trajectoryController:
    def __init__(self, klp=0.2, klv=0.5, kli=0.2, kap=0.8, kav=0.6, kai=0.2):
        self.PID_linear=PID_ctrl(kp=klp, kv=klv, ki=kli,)
        self.PID_angular=PID_ctrl(kp=kap, kv=kav, ki=kai)
    
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
