import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import numpy as np
import time
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension, UInt8MultiArray
from math import atan2, sqrt, sin, cos, pi as M_PI
import numpy as np
import math

def create_yaw_from_quaternion(quaternion):
    return atan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y), 1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2))

def create_quaternion_from_yaw(yaw):
    return Quaternion(
        x=0.0,
        y=0.0,
        z=sin(yaw / 2.0),
        w=cos(yaw / 2.0)
    )

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ]

def calculate_positioning_error(from_pose, current_pose, goal):
    displacement = np.sqrt((from_pose.position.x - current_pose.position.x) ** 2 + (from_pose.position.y - current_pose.position.y) ** 2)
    
    if goal > 0:
        return goal - displacement
    else:
        return -(abs(goal) - displacement)

def calculate_linear_error(pose, goal):
    linear_error = sqrt((goal[0] - pose.position.x) ** 2 + (goal[1] - pose.position.y) ** 2)
    return linear_error

def calculate_rotation_error(pose, goal):
    angular_error = goal[2] - pose.orientation.z
    return normalize_angle(angular_error)

def calculate_angular_error(pose, goal):
    angular_error = atan2(goal[1] - pose.position.y, goal[0] - pose.position.x) - pose.orientation.z
    return normalize_angle(angular_error)

def normalize_angle(angle):
    if angle <= -M_PI:
        angle += 2*M_PI
    elif angle >= M_PI:
        angle -= 2*M_PI
    return angle

class JointOdomSubscriber(Node):
    def __init__(self):
        super().__init__('joint_odom_subscriber')

        # Subscribe to /joint_states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)

        # Subscribe to /odom
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        # Kinematics model
        self.kinematics = DifferentialDriveKinematics(
            wheel_radius=0.033,  # TurtleBot3 wheel radius (m)
            # wheel_separation=0.16,  # TurtleBot3 track width (m)
            wheel_separation=0.08,
            ticks_per_rev=4096  # Encoder resolution
        )

        self.current_pose = None

    def joint_state_callback(self, msg):
        """ Extracts wheel positions from /joint_states """
        left_wheel = 'wheel_left_joint'
        right_wheel = 'wheel_right_joint'

        try:
            if left_wheel in msg.name and right_wheel in msg.name:
                left_idx = msg.name.index(left_wheel)
                right_idx = msg.name.index(right_wheel)

                left_rads = msg.position[left_idx]  # Encoder value
                right_rads = msg.position[right_idx]  # Encoder value

                # Get current time
                current_time = time.time()

                # Compute kinematics
                result = self.kinematics.update(left_rads, right_rads, current_time)
                if result:
                    x, y, theta = result
                    self.get_logger().info(f'[JOINT] x: {x:.3f}, y: {y:.3f}, θ: {theta:.3f}')

        except Exception as e:
            self.get_logger().error(f'Error processing joint states: {e}')

    def odom_callback(self, msg):
        """ Extracts pose (x, y, θ) from /odom """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        theta = self.quaternion_to_euler(quat)

        self.current_pose = (x, y, theta)
        self.get_logger().info(f'[ODOM] x: {x:.3f}, y: {y:.3f}, θ: {theta:.3f}')

    def quaternion_to_euler(self, quat):
        return create_yaw_from_quaternion(quat)
        """ Converts quaternion to yaw angle (θ) """
        qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy**2 + qz**2)
        return np.arctan2(siny_cosp, cosy_cosp)


class DifferentialDriveKinematics:
    def __init__(self, wheel_radius, wheel_separation, ticks_per_rev):
        self.r = wheel_radius
        self.L = wheel_separation
        self.ticks_per_rev = ticks_per_rev

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.prev_left_rads = None
        self.prev_right_rads = None
        self.prev_time = None

    def update(self, left_rads, right_rads, current_time):
        if self.prev_left_rads is None or self.prev_right_rads is None:
            self.prev_left_rads = left_rads
            self.prev_right_rads = right_rads
            self.prev_time = current_time
            return None

        dt = current_time - self.prev_time
        if dt == 0:
            return None

        delta_left_rads = left_rads - self.prev_left_rads
        delta_right_rads = right_rads - self.prev_right_rads

        d_L = self.r * delta_left_rads
        d_R = self.r * delta_right_rads
        d_C = (d_L + d_R) / 2
        delta_theta = (d_R - d_L) / self.L

        if abs(delta_theta) < 1e-6:  # Approximate straight-line motion
            self.x += d_C * math.cos(self.theta)
            self.y += d_C * math.sin(self.theta)
        else:  # Follow circular motion
            R = d_C / delta_theta  # Instantaneous radius of curvature
            self.x += R * (math.sin(self.theta + delta_theta) - math.sin(self.theta))
            self.y -= R * (math.cos(self.theta + delta_theta) - math.cos(self.theta))

        self.theta += delta_theta
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi  # Normalize theta to [-pi, pi]

        # Update previous values
        self.prev_left_rads = left_rads
        self.prev_right_rads = right_rads
        self.prev_time = current_time

        return self.x, self.y, self.theta

        # # theta_L = (delta_L / self.ticks_per_rev) * 2 * np.pi
        # # theta_R = (delta_R / self.ticks_per_rev) * 2 * np.pi
        # w_left = delta_L / dt
        # w_right = delta_R / dt

        # # v = (d_L + d_R) / (2 * dt)
        # # w = (d_R - d_L) / (self.L * dt)

        # # self.theta += w * dt
        # # self.x += v * np.cos(self.theta) * dt
        # # self.y += v * np.sin(self.theta) * dt

        # self.prev_left_ticks = left_ticks
        # self.prev_right_ticks = right_ticks
        # self.prev_time = current_time

        # # return v, w, self.x, self.y, self.theta
        # wheel_velocities = np.array([[w_left], 
        #                    [w_right]])

        # transform = np.array([[self.L,      self.L],
        #                       [0.0,         0.0],
        #                       [-1.0,        1.0]])
        # scaling = self.r / (2*self.L)
        # displacements = scaling * (transform @ wheel_velocities)

        # vel_x_robot = displacements[0][0]
        # w = displacements[2][0]

        # vel_x = vel_x_robot * math.cos(self.theta)
        # vel_y = vel_x_robot * math.sin(self.theta)
        
        # delta_x = vel_x * dt
        # delta_y = vel_y * dt
        # delta_theta = w * dt

        # new_yaw = self.theta + delta_theta
        # new_yaw = normalize_angle(new_yaw)

        # self.x += delta_x
        # self.y += delta_y
        # self.theta = new_yaw

        # return self.x, self.y, new_yaw
        # # imu hack
        # if rp.filter_type == rp.FilterType.ODOMETRY_IMU:
        #     new_yaw = create_yaw_from_quaternion(self.imu.orientation)
        #     delta_theta = new_yaw - old_yaw
        #     w = delta_theta / dt

        # self.odom.header.stamp = stamp.to_msg()
        # self.odom.pose.pose.position.x += delta_x
        # self.odom.pose.pose.position.y += delta_y
        # self.odom.twist.twist.linear.x = vel_x
        # self.odom.twist.twist.linear.y = vel_y
        # self.odom.pose.pose.orientation = create_quaternion_from_yaw(new_yaw)
        # self.odom.twist.twist.angular.z = w

        # self.odom_pub.publish(self.odom)


def main(args=None):
    rclpy.init(args=args)
    node = JointOdomSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
