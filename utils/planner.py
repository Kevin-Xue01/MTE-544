import math
from math import atan, atan2, cos, sin, pi
import numpy as np
from enum import Enum, auto

class PathType(Enum):
    CIRCLE = auto()
    ZIGZAG = auto()
    SPORADIC = auto()
    SQUARE = auto()
    SNAKE = auto()


class planner:
    def __init__(self, _type: PathType):
        if _type == PathType.CIRCLE:
            self.traj = self.generate_circular_path()
        elif _type == PathType.ZIGZAG:
            self.traj = self.generate_zigzag_path()
        elif _type == PathType.SPORADIC:
            self.traj = self.generate_spuratic_path()
        elif _type == PathType.SQUARE:
            self.traj = self.generate_square_path()
        elif _type == PathType.SNAKE:
            self.traj = self.generate_snake_path()
        else:
            raise ValueError(f"Unknown path type")
        
        print(self.traj)

    def generate_circular_path(self):
        center = (0, 2)
        radius = 2
        
        angles = np.linspace(-np.pi/2, 2.9*np.pi/2, 100)
        
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        return list(zip(x, y))
    
    def generate_zigzag_path(self):
        path = []
        for i in range(10):
            x = i
            y = 2 if i % 2 == 0 else -2
            path.append((x, y))
        return path

    def generate_spuratic_path(self):
        np.random.seed(42)
        x = np.random.uniform(-5, 5, 20)
        y = np.random.uniform(-5, 5, 20)
        return list(zip(x, y))

    def generate_square_path(self):
        side_length = 4
        path = []
        num_points_per_side = 4
        # Bottom side
        for i in range(num_points_per_side + 1):
            path.append((i * side_length / num_points_per_side, 0))
        # Right side
        for i in range(1, num_points_per_side + 1):
            path.append((side_length, i * side_length / num_points_per_side))
        # Top side
        for i in range(1, num_points_per_side + 1):
            path.append((side_length - i * side_length / num_points_per_side, side_length))
        # Left side
        side_length = 2
        for i in range(1, num_points_per_side):
            path.append((0, side_length - i * side_length / num_points_per_side))
        return path

    # def generate_out_and_back_path(self):
    #     path = []
    #     for i in range(5):
    #         path.append((i, 0))
    #     for i in range(5, 3, -1):
    #         path.append((i, 0))
    #     return path

    def generate_snake_path(self):
        path = []
        x, y = 0, 0  # Starting point
        step = 5  # Initial step size
        while step > 0:
            # Move right
            for _ in range(step):
                path.append((x, y))
                x += 1
            step -= 1
            # Move up
            for _ in range(step):
                path.append((x, y))
                y += 1
            # Move left
            for _ in range(step):
                path.append((x, y))
                x -= 1
            step -= 1
            # Move down
            for _ in range(step):
                path.append((x, y))
                y -= 1
        return path
