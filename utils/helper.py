import csv
from math import asin, atan2, sqrt, pi

class CSVLogger:
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = headers
        
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)

    def log(self, data):
        """Append a row of data to the CSV file."""
        if len(data) != len(self.headers):
            raise ValueError("Data length does not match headers length.")
        
        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    
    
def euler_from_quaternion(quat):
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
    return yaw


def calculate_linear_error(current_pose, goal_pose):
    return sqrt( (current_pose[0] - goal_pose[0])**2 +
                (current_pose[1] - goal_pose[1])**2 )

def calculate_angular_error(current_pose, goal_pose):

    error_angular= atan2(goal_pose[1]-current_pose[1],
                        goal_pose[0]-current_pose[0]) - current_pose[2]
    
    if error_angular <= -pi:
        error_angular += 2*pi
    
    
    elif error_angular >= pi:
        error_angular -= 2*pi
    
    return error_angular