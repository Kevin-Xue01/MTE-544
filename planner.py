import math
from math import atan, atan2, cos, sin, pi
import numpy as np


class planner:
    def __init__(self, type="circular"):
        self.type = type
        if self.type == "circular":
            self.traj = self.generate_circular_path()
        elif self.type == "zigzag":
            self.traj = self.generate_zigzag_path()
        elif self.type == "sporadic":
            self.traj = self.generate_spuratic_path()
        elif self.type == "square":
            self.traj = self.generate_square_path()
        elif self.type == "out_and_back":
            self.traj = self.generate_out_and_back_path()
        else:
            raise ValueError(f"Unknown path type: {self.type}")

    def generate_circular_path(self):
        # Define the center and radius of the circle
        center = (0, 2)
        radius = 2
        
        # Define angles corresponding to the given points
        angles = np.linspace(-np.pi/2, 2.9*np.pi/2, 100)  # From (0,0) counterclockwise
        
        # Compute the x, y coordinates of the circle
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        return list(zip(x, y))
    
    def generate_zigzag_path(self):
        # Generate a zigzag trajectory
        path = []
        for i in range(10):
            x = i
            y = 2 if i % 2 == 0 else -2
            path.append((x, y))
        return path

    def generate_spuratic_path(self):
        # Generate a sporadic trajectory with random points
        np.random.seed(42)  # For reproducibility
        x = np.random.uniform(-5, 5, 20)
        y = np.random.uniform(-5, 5, 20)
        return list(zip(x, y))

    def generate_square_path(self):
        # Generate a square trajectory
        side_length = 4
        path = [
            (side_length, 0),
            (side_length, side_length),
            (0, side_length),
            (0, 0.5)  # Closing the square
        ]
        return path

    def generate_out_and_back_path(self):
        # Generate an out-and-back trajectory
        path = []
        for i in range(5):
            path.append((i, 0))  # Outward path
        for i in range(5, -1, -1):
            path.append((i, 0))  # Backward path
        return path