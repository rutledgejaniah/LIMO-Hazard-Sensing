#!/usr/bin/python

import rospy
from nav_msgs.msg import Odometry
# from fire_detection.msg import ObjectDetectorData
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import math
import time
from std_msgs.msg import Int32


class PID:
    """PID Controller
    """

    def __init__(self, P=0.0, I=0.0, D=0.0, current_time=None):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback"""
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if delta_time >= self.sample_time:
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if self.ITerm < -self.windup_guard:
                self.ITerm = -self.windup_guard
            elif self.ITerm > self.windup_guard:
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            return self.output

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup"""
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval."""
        self.sample_time = sample_time


class VehicleState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def update_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta


class OdometryEstimator:
    def __init__(self):
        rospy.init_node('odometry_estimator')

        # Publishers and Subscribers
        self.cmd_vel_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.imagine_subscriber = rospy.Subscriber("imagine_data", Int32, self.handle_obj_detector_data_imagine)
        self.distance_subscriber = rospy.Subscriber("distance_data", Int32, self.handle_obj_detector_data_distance)
        self.odom_subscriber = rospy.Subscriber("odom", Odometry, self.odom_callback, queue_size=10)

        # Timer for control loop
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

        # Robot State
        self.state_ = VehicleState()

        # Initialize coordinates from object detection data
        self.a = 0.0  # X coordinate of waypoint
        self.b = 0.0  # Y coordinate of waypoint

        # PID controllers
        self.pid_linear = PID(-0.1, 0.000, 0.0000)
        self.pid_angular = PID(-0.8, 0.0, 0.0)
        self.max_linear_velocity = 2.0
        self.max_angular_velocity = 8.0

        self.integral_angle = 0.0
        self.previous_angle_error = 0.0

        # List of goal poses
        self.goal_poses = [{'x': self.a, 'y': self.b, 'theta': 0.0}]
        self.current_goal_index = 0
    
    def update_goal_poses(self):
        # Update goal poses based on the new values of self.a and self.b
        self.goal_poses = [{'x': self.a, 'y': self.b, 'theta': 0.0},
                           {'x': self.a, 'y': self.b, 'theta': 0.0},
                           {'x': self.a, 'y': self.b, 'theta': 0.0}]

    def handle_obj_detector_data_imagine(self, imagine):
        # Print imaginary line distance from FireDetectionNode
        rospy.loginfo("Received imagine line: %d", imagine.data)
        # Extract vertical distance 
        self.a = imagine.data
        self.update_goal_poses()

    def handle_obj_detector_data_distance(self, distance):
        # Print distance to object from FireDetectionNode
        rospy.loginfo("Received distance: %d", distance.data)
        # Horozontial distance calculation 
        self.b = math.sqrt(abs(distance.data**2 - self.a**2))
        self.update_goal_poses()
         

    def odom_callback(self, msg):
        # Extract robot pose from odometry message
        pose = msg.pose.pose
        orientation_q = pose.orientation
        _, _, yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.state_.update_state(pose.position.x, pose.position.y, yaw)

    def timer_callback(self, event):
        self.drive_to_goal()

    def drive_to_goal(self):
        print("x: {}, y: {}".format(self.state_.x, self.state_.y))
        if self.current_goal_index < len(self.goal_poses):
            current_goal = self.goal_poses[self.current_goal_index]
            dx = current_goal['x'] - self.state_.x
            dy = current_goal['y'] - self.state_.y
            print("x-goal: {}, y-goal: {}".format(current_goal['x'], current_goal['y']))
            distance_to_goal = math.sqrt(dx**2 + dy**2)
            print("distance_to_goal: {}".format(distance_to_goal))

            distance_threshold = 1.0  # Meters
            if distance_to_goal < distance_threshold:
                rospy.loginfo("Goal pose {} reached.".format(self.current_goal_index))
                self.current_goal_index += 1
                if self.current_goal_index < len(self.goal_poses):
                    rospy.loginfo("Driving to goal pose {}.".format(self.current_goal_index))
                else:
                    rospy.loginfo("All goal poses reached.")
                    self.cmd_vel_publisher.publish(Twist())  # Stop the robot
                return

            target_angle = math.atan2(dy, dx)
            angular_error = target_angle - self.state_.theta
            print("angle_error: {}".format(angular_error))
            linear_error = distance_to_goal

            linear_velocity = self.pid_linear.update(linear_error)
            angular_velocity = self.pid_angular.update(angular_error)

            print("linear_velocity: {}".format(linear_velocity))
            print("angular_velocity: {}".format(angular_velocity))

            # Limit velocities
            linear_velocity = min(linear_velocity, self.max_linear_velocity)
            angular_velocity = max(min(angular_velocity, self.max_angular_velocity), -self.max_angular_velocity)

            # Publish command velocities
            twist_msg = Twist()
            twist_msg.linear.x = linear_velocity
            twist_msg.angular.z = angular_velocity
            self.cmd_vel_publisher.publish(twist_msg)

def main():
    try:
        odometry_estimator = OdometryEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
