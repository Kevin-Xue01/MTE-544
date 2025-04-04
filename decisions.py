import sys
from enum import Enum, auto

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry as odom
from rclpy import init, spin, spin_once
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from utils import PathType, Planner
from utils.controller import Controller
from utils.helper import (
    calculate_angular_error,
    calculate_linear_error,
    euler_from_quaternion,
)
from utils.localization import Localization


class decision_maker(Node):
    def __init__(self):
        super().__init__("decision_maker")
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=2, history=1, depth=10)
        self.publisher=self.create_publisher(Twist, "/cmd_vel", qos_profile=qos)
        
        self.rate = 10
        self.publishing_period = 1 / self.rate
        self.reachThreshold = 0.05

        self.localizer = Localization()
        
        self.controller = Controller()
        self.planner = Planner()
        
        self.create_timer(self.publishing_period, self.timerCallback)

    def timerCallback(self):
        spin_once(self.localizer)

        if self.localizer.getPose() is  None:
            print("waiting for odom msgs ....")
            return
        
        vel_msg = Twist()

        reached_goal = True if calculate_linear_error(self.localizer.getPose(), self.planner.traj[-1]) < self.reachThreshold else False

        if reached_goal:
            print("reached goal")
            self.publisher.publish(vel_msg)
            raise SystemExit
        
        velocity, yaw_rate = self.controller.vel_request(self.localizer.getPose(), self.planner.traj)

        vel_msg.linear.x = velocity
        vel_msg.angular.z = yaw_rate
        
        self.publisher.publish(vel_msg)


def main(args=None):
    init()
    DM = decision_maker()

    try:
        spin(DM)
    except SystemExit:
        print(f"reached there successfully {DM.localizer.pose}")

if __name__=="__main__":
    main()
