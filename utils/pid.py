from rclpy.time import Time

from .config import _config
from .constants import ControllerType


class PID:
    def __init__(self, kp, kv, ki):
        self.history_length = 3
        self.history = []

        self.kp = kp
        self.kv = kv
        self.ki = ki
        
    def update(self, stamped_error, status):
        if status == False:
            self.__update(stamped_error)
            return 0,0
        else:
            return self.__update(stamped_error)

    def __update(self, stamped_error):
        
        latest_error = stamped_error[0]
        stamp = stamped_error[1]
        
        self.history.append(stamped_error)        
        
        if (len(self.history) > self.history_length):
            self.history.pop(0)
        
        # If insufficient data points, use only the proportional gain
        if (len(self.history) != self.history_length):
            return self.kp * latest_error
        
        dt_avg = 0
        error_dot = 0
        
        for i in range(1, len(self.history)):
            t0 = Time.from_msg(self.history[i-1][1])
            t1 = Time.from_msg(self.history[i][1])
            
            dt = (t1.nanoseconds - t0.nanoseconds) / 1e9
            
            dt_avg += dt            
            
            dt = 0.1
            error_dot += (self.history[i][0] - self.history[i-1][0])/dt
            
        error_dot /= len(self.history)
        dt_avg /= len(self.history)
        
        sum_ = 0
        for hist in self.history:
            sum_ += hist[0] 
        
        error_int = sum_*dt_avg
        
        if _config.controller_type == ControllerType.P:
            return self.kp * latest_error
        
        elif _config.controller_type == ControllerType.PD:
            return self.kp * latest_error + self.kv * error_dot
        
        elif _config.controller_type == ControllerType.PI:
            return self.kp * latest_error +  self.ki * error_int
        
        elif _config.controller_type == ControllerType.PID:
            return self.kp * latest_error + self.kv * error_dot + self.ki * error_int