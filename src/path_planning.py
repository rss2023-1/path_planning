#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.odom_topic = "pf/pose/odom" #rospy.get_param("~odom_topic") change this back to param eventually
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        self.map = None
        self.cur_pos = None
        self.res = None
        self.ori = None
        self.pos = None

    def convert_pix_to_real(self, u, v):
        u1 = u*self.res
        v1 = v*self.res
        x = u1 + self.pos[0]
        y = v1 + self.pos[1]
        return x, y

    def convert_real_to_pix(self, x, y):
        x1 = x - self.pos[0]
        y1 = y - self.pos[1]
        u = x1 / self.res
        v = y1 / self.res
        return u, v


    def map_cb(self, msg):
        data = msg.data
        self.res = msg.info.resolution
        self.ori = msg.info.origin.orientation
        self.pos = msg.info.origin.position
        self.wid =  msg.info.width
        self.hei =  msg.info.height

        """
        print("d", data)
        print("d len1", type(data), len(data))
        
        print("r", res)
        print("o", ori)
        print("p", pos) 
        """

        arr = np.array(data)
        arr = np.reshape(arr, (self.hei, self.wid))

        self.map = arr
        


    def odom_cb(self, msg):
        self.cur_pos = msg
        print(msg)


    def goal_cb(self, msg):
        self.plan_path(self.cur_pos, msg, self.map)

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##

        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
