import math
from math import atan, atan2, cos, sin, pi
import numpy as np


class planner:
    def __init__(self):
        self.traj = self.generate_circular_path()

    def generate_circular_path(self):
        # Define the center and radius of the circle
        center = (0, 2)
        radius = 2
        
        # Define angles corresponding to the given points
        angles = np.linspace(-np.pi/2, 2.95*np.pi/2, 100)  # From (0,0) counterclockwise
        
        # Compute the x, y coordinates of the circle
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        return list(zip(x, y))