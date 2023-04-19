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
        self.odom_topic       = rospy.get_param("~odom_topic")
        self.lookahead        = rospy.get_param("~lookahead_dist")
        self.speed            = rospy.get_param("~pursuit_speed")
        self.wheelbase_length = 0.2 #measure this for sure
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

        # might want to extract pose directions as well? or is just x,y sufficient
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

    def lookahead_intersection(self, pose, segment):
        """
        Takes in pose as Odometry msg and a line segment index in the list of 
        poses for the trajectory. Returns intersection point of the closest trajectory
        that intersects with a circle of radius `self.lookahead_dist` centered on
        pose.
        Intersection equation: 
        https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428
        """
        center = np.array(([pose.pose.pose.position.x, pose.pose.pose.position.y]))
        r = self.lookahead
        intersection = False
        current_segment = segment
        # We iterate through subsequent line segments after closest until intersection
        while (not intersection and current_segment < self.x_values.size - 1):
            seg_start = np.array([self.x_values[current_segment], self.y_values[current_segment]])
            seg_v = np.array([self.x_values[current_segment+1], self.y_values[current_segment+1]]) - seg_start
            current_segment += 1
            a = np.dot(seg_v, seg_v)
            b = 2 * np.dot(seg_v, seg_start - center)
            c = np.dot(seg_start, seg_start) + np.dot(center, center) - 2 * np.dot(seg_start, center) - r**2
            disc = b**2 - 4 * a * c
            if disc < 0:
                # no solution, continue to next segment
                continue
            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            if not (0 <= t1 <= 1 or 0 <= t2 <= 1):
                # intersection outside of line segment, continue
                continue
            soln_param = t1
            if (0 <= t1 <= 1 and 0 <= t2 <= 1):
                # edge case, currently return t1
                print("edge case!")
            else (0 <= t2 <= 1):
                # t2 is our intersection point
                soln_param = t2
        return seg_start + soln_param * seg_v
        









        


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
