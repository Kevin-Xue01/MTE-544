from math import asin, atan2, sqrt

M_PI=3.1415926535
import csv
import logging
from enum import Enum, auto

import numpy as np


class LocalizationMode(Enum):
    RAW = auto()
    EKF = auto()
    UKF = auto() 

class CSVLogger:
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = headers
        
        # Create and write headers if file is empty
        try:
            with open(self.filename, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)
        except FileExistsError:
            pass  # File already exists, don't overwrite headers

    def log(self, data):
        """Append a row of data to the CSV file."""
        if len(data) != len(self.headers):
            raise ValueError("Data length does not match headers length.")
        
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    
    
# Conversion from Quaternion to Euler Angles
def euler_from_quaternion(quat):
    """
    Convert quaternion (w in last place) to euler roll, pitch, yaw.
    quat = [x, y, z, w]
    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)
    # just unpack yaw for tb
    return yaw


# Calculation of the linear error
def calculate_linear_error(current_pose, goal_pose):
    return sqrt( (current_pose[0] - goal_pose[0])**2 +
                (current_pose[1] - goal_pose[1])**2 )

# Calculation of the angular error
def calculate_angular_error(current_pose, goal_pose):

    error_angular= atan2(goal_pose[1]-current_pose[1],
                        goal_pose[0]-current_pose[0]) - current_pose[2]
    
    if error_angular <= -M_PI:
        error_angular += 2*M_PI
    
    
    elif error_angular >= M_PI:
        error_angular -= 2*M_PI
    
    return error_angular