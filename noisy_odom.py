# from copy import deepcopy

# import numpy as np
# import rclpy
# from nav_msgs.msg import Odometry
# from rclpy.node import Node

# from utils import CSVLogger

# """
# Ornstein-Uhlenbeck (OU) Noise Process Explanation:

# This noise method uses an Ornstein-Uhlenbeck process to generate time-correlated noise
# for the twist (velocity) measurements. The OU process is defined by the equation:

#     x_next = x_current + theta * (mu - x_current) * dt + sigma * sqrt(dt) * N(0, 1)

# where:
#     - theta is the rate of mean reversion (how quickly the noise reverts toward the mean),
#     - mu is the long-term mean (set to 0 for no sustained bias),
#     - sigma is the volatility (magnitude of the noise),
#     - dt is the time step, and
#     - N(0, 1) is a standard normal random variable.

# OU noise adds time-correlated errors that simulate persistent biases (e.g., wheel slip).
# Unlike white noise, these errors evolve over time, forcing the filter to continuously adapt.
# This results in a more realistic sensor simulation.
# """

# class NoisyOdometry(Node):
#     def __init__(self):
#         super().__init__("noisy_odometry")

#         self.odom_sub = self.create_subscription(
#             Odometry, "/odom", self.odom_callback, 10)

#         self.odom_pub = self.create_publisher(Odometry, "/noisy_odom", 10)

#         self.theta = 0.15
#         self.mu = 0.0
#         self.dt = 0.1
#         self.sigma_linear = 0.1
#         self.sigma_angular = 0.05

#         self.ou_noise_linear = 0.0
#         self.ou_noise_angular = 0.0

#         self.get_logger().info("Noisy Odometry Node Started (using OU process noise)")

#     def ornstein_uhlenbeck(self, x, theta, mu, sigma, dt):
#         return x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.randn()

#     def odom_callback(self, msg):
#         raw_x = msg.pose.pose.position.x
#         raw_y = msg.pose.pose.position.y
#         raw_v = msg.twist.twist.linear.x
#         raw_w = msg.twist.twist.angular.z

#         self.ou_noise_linear = self.ornstein_uhlenbeck(
#             self.ou_noise_linear, self.theta, self.mu, self.sigma_linear, self.dt)
#         self.ou_noise_angular = self.ornstein_uhlenbeck(
#             self.ou_noise_angular, self.theta, self.mu, self.sigma_angular, self.dt)

#         noisy_msg = Odometry()
#         noisy_msg.header = msg.header
#         noisy_msg.pose.pose = msg.pose.pose

#         noisy_twist = deepcopy(msg.twist.twist)
#         noisy_twist.linear.x += self.ou_noise_linear
#         noisy_twist.angular.z += self.ou_noise_angular
#         noisy_msg.twist.twist = noisy_twist
        
#         self.odom_pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = NoisyOdometry()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()

import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointOdomSubscriber(Node):
    def __init__(self):
        super().__init__('joint_odom_subscriber')

        # Subscribe to /joint_states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        self.odom_pub = self.create_publisher(Odometry, "/noisy_odom", 10)


        # TurtleBot3 Parameters
        self.ticks_per_rev = 4096  # Encoder resolution
        self.kinematics = DifferentialDriveKinematics(
            wheel_radius=0.033,  # TurtleBot3 wheel radius (m)
            wheel_separation=0.16,  # TurtleBot3 track width (m)
            ticks_per_rev=self.ticks_per_rev
        )

        self.current_pose = None

    def joint_state_callback(self, msg):
        """ Extracts wheel positions (converted from radians to ticks) from /joint_states """
        left_wheel = 'wheel_left_joint'
        right_wheel = 'wheel_right_joint'

        try:
            if left_wheel in msg.name and right_wheel in msg.name:
                left_idx = msg.name.index(left_wheel)
                right_idx = msg.name.index(right_wheel)

                # Convert wheel positions from radians to encoder ticks
                left_ticks = (msg.position[left_idx] * self.ticks_per_rev) / (2 * np.pi)
                right_ticks = (msg.position[right_idx] * self.ticks_per_rev) / (2 * np.pi)

                # Get current time
                current_time = time.time()

                # Compute kinematics
                result = self.kinematics.update(left_ticks, right_ticks, current_time)
                if result:
                    v, w, x, y, theta = result
                    # self.get_logger().info(f'[JOINT] v: {v:.3f}, w: {w:.3f}, x: {x:.3f}, y: {y:.3f}, θ: {theta:.3f}')
                    msg = Odometry()
                    msg.pose.pose.position.x = x 
                    msg.pose.pose.position.y = y
                    msg.twist.twist.linear.x = v
                    msg.twist.twist.angular.z = w

                    # self.ou_noise_linear = self.ornstein_uhlenbeck(
                    #     self.ou_noise_linear, self.theta, self.mu, self.sigma_linear, self.dt)
                    # self.ou_noise_angular = self.ornstein_uhlenbeck(
                    #     self.ou_noise_angular, self.theta, self.mu, self.sigma_angular, self.dt)

                    # noisy_msg = Odometry()
                    # noisy_msg.header = msg.header
                    # noisy_msg.pose.pose = msg.pose.pose

                    # noisy_twist = deepcopy(msg.twist.twist)
                    # noisy_twist.linear.x += self.ou_noise_linear
                    # noisy_twist.angular.z += self.ou_noise_angular
                    # noisy_msg.twist.twist = noisy_twist
                    
                    self.odom_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error processing joint states: {e}')

class DifferentialDriveKinematics:
    def __init__(self, wheel_radius, wheel_separation, ticks_per_rev):
        self.r = wheel_radius
        self.L = wheel_separation
        self.ticks_per_rev = ticks_per_rev

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.prev_left_ticks = None
        self.prev_right_ticks = None
        self.prev_time = None

    def update(self, left_ticks, right_ticks, current_time):
        if self.prev_left_ticks is None or self.prev_right_ticks is None:
            self.prev_left_ticks = left_ticks
            self.prev_right_ticks = right_ticks
            self.prev_time = current_time
            return None

        dt = current_time - self.prev_time
        if dt == 0:
            return None

        delta_L = left_ticks - self.prev_left_ticks
        delta_R = right_ticks - self.prev_right_ticks

        theta_L = (delta_L / self.ticks_per_rev) * 2 * np.pi
        theta_R = (delta_R / self.ticks_per_rev) * 2 * np.pi
        d_L = self.r * theta_L
        d_R = self.r * theta_R

        v = (d_L + d_R) / (2 * dt)
        w = (d_R - d_L) / (self.L * dt)

        self.theta += w * dt
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt

        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks
        self.prev_time = current_time

        return v, w, self.x, self.y, self.theta


def main(args=None):
    rclpy.init(args=args)
    node = JointOdomSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()