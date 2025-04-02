import math
from math import atan, atan2, cos, sin


class planner:
    def __init__(self, radius=1.0, num_points=30):
        self.traj = [
            [radius * cos(2 * pi * i / num_points), radius * sin(2 * pi * i / num_points)]
            for i in range(num_points)
        ]
