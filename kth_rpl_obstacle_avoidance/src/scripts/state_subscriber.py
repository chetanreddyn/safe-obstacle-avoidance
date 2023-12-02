#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
# from your_msgs.msg import SafeSetConfig 
import tf.transformations
import numpy as np


 

def publish_velocity_commands():
    rospy.init_node('velocity_publisher', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    rate = rospy.Rate(100)  # 10 Hz (adjust the frequency as needed)

    while not rospy.is_shutdown():
        cmd = Twist()
        cmd.linear.x = -0.1  # Linear velocity (m/s)
        cmd.angular.z = 0  # Angular velocity (radians/s)
        pub.publish(cmd)
    

        rate.sleep()

class Controller:
    def __init__(self,robot_config):
        rospy.Subscriber('/odom',Odometry,self.odom_callback)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.theta_min  = robot_config['theta_min']
        self.theta_max  = robot_config['theta_max']
        self.max_depth  = robot_config['max_depth']
        self.min_depth  = robot_config['min_depth']
        self.r_limit    = robot_config['r_limit']
        self.r0         = robot_config['r0']
        self.l          = robot_config['l']
        # rospy.Subscriber('/safe_set_config',SafeSetConfig,update_safeset_callback)

    def vel_publish(self,info=None):
        cmd = Twist()
        cmd.linear.x = -0.1  # Linear velocity (m/s)
        cmd.angular.z = 0  # Angular velocity (radians/s)
        self.pub.publish(cmd)
        # print("Published")

    def odom_callback(self,data):
        pose = data.pose.pose
        linear_velocity = data.twist.twist.linear
        angular_velocity = data.twist.twist.angular

        quaternion = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
        # Convert the quaternion to a rotation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(quaternion)

        # Extract the yaw angle (rotation about Z-axis)
        assert self.l<0 # Remove the 180 degree
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])*180/np.pi + 180
        if yaw>180:
            yaw = yaw-360

        print("(x, y, theta): {:.2f}, {:.2f}, {:.2f}, (u,w): {:.2f},{:.2f}".format(pose.position.x, pose.position.y, yaw,linear_velocity.x,angular_velocity.z))


def main():
    rospy.init_node("state_subscriber")

    robot_config = {'theta_min':2*np.pi/6, # Minimum Angle of the Camera
                'theta_max':4*np.pi/6, # Maximum Angle of the Camera
                'max_depth':2.5, # Maximum Depth Perceived by the Camera (Used in the Loss Function)
                'min_depth':0.3, # Minimum Depth Perceived by the Camera
                'r_limit'  :2.5, # Maximum Radial Distance to Depict in the Plots
                'r0'       :0.15, # Robot Radius 
                'l'        :-0.08
                # 'resolution': 1920
                }

    controller_obj = Controller(robot_config=robot_config)

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        # print('here')
        current_time = rospy.get_rostime()

        # Print the current ROS time
        # print("Current ROS Time: %f seconds" % current_time.to_sec())

        # controller_obj.vel_publish()
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
