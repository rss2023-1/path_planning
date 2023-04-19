#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = '/pf/pose/odom' # rospy.get_param("~odom_topic")
        self.lookahead        = 1 # FILL IN #
        self.speed            = 0.5 # FILL IN #
        self.wheelbase_length = 0.5 # FILL IN #
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.traj_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.x_values = []
        self.y_values = []


    def distance(self, P0, P1, p):
        # Calculate the vector T and V
        T = P1 - P0
        V = p - P0
        # Calculate the scalar L and U using dot product
        L = np.sum(T**2, axis=1)
        U = np.sum(T*V, axis=1)/L
        # Clip U to ensure it is within the bounds of [0,1]
        U = np.clip(U, 0, 1)
        # Calculate C using vector T and scalar U
        C = P0 + U[:,np.newaxis]*T
        # Calculate the distance between C and p using norm function
        return np.linalg.norm(C - p, axis=1)


    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print "Receiving new trajectory:", len(msg.poses), "points"
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            self.x_values.append(x)
            self.y_values.append(y)

        self.x_values = np.array(self.x_values)
        self.y_values = np.array(self.y_values)

        


    def odometry_callback(self, odom):
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        p = np.array([px, py])
        print('odom', odom)
        print('x_values', self.x_values)
        print('y_values', self.y_values)
        np.array([self.distance(self.x_values,self.y_values,p_i) for p_i in p])

        


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
