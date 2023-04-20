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
        self.odom_topic       = rospy.get_param("~odom_topic", "pf/pose/odom")
        self.lookahead        = rospy.get_param("~lookahead_dist", 0.5)
        self.speed            = rospy.get_param("~pursuit_speed", 0.5)
        self.wheelbase_length = 0.2 #measure this for sure
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.traj_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.x_values = []
        self.y_values = []


    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print "Receiving new trajectory:", len(msg.poses), "points"
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        # Read trajectory x and y values when initialized
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            self.x_values.append(x)
            self.y_values.append(y)

        self.x_values = np.array(self.x_values)
        self.y_values = np.array(self.y_values)


    def odometry_callback(self, odom):
        # Get path coords from traj initialization
        path_coords = np.vstack((self.x_values, self.y_values))
        # Initialize vector containing all line segment points
        pairs = np.zeros((2, 2*(path_coords.shape[1]-1)))
        pairs[:, ::2] = path_coords[:, :-1]
        pairs[:, 1::2] = path_coords[:, 1:]

        # pairs = [p1, p2, p2, p3, p3, p4, ...] where p is 2x1 [x y]'
        # for indexing, want line_segs = [[p1, p2], [p2, p3], [p3, p4], ...]

        # Group pairs into line segments
        pairst = np.transpose(pairs)
        line_segs = pairst.reshape(-1, 2, 2)

        # Get baselink position as x, y coord
        xpos = odom.pose.pose.position.x
        ypos = odom.pose.pose.position.y
        point = np.array([xpos, ypos])
        
        # Calculate minimum distances between current position and each line segment
        # Initialize array
        min_distances = np.zeros(line_segs.shape[0])
        # Loop over each line segment
        for i in range(line_segs.shape[0]):
            # Get the starting and ending points of the line segment
            p1 = line_segs[i, 0]
            p2 = line_segs[i, 1]
            # Compute the vector representing the line segment
            v = p2 - p1
            # Compute the vector from the starting point of the line segment to the point
            w = point - p1
            # Compute the projection of w onto v
            proj = np.dot(w, v) / np.dot(v, v) * v
            # Compute the distance between the point and the projected point
            dist = np.linalg.norm(proj - w)
            # If the projection is outside the line segment, compute the distance to the closest endpoint
            if np.dot(proj - p1, proj - p2) > 0:
                dist = min(np.linalg.norm(point - p1), np.linalg.norm(point - p2))
            # Store the minimum distance for this line segment
            min_distances[i] = dist
        
        print('min_dists', min_distances)

        


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
