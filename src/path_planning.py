#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from heapdict import *
from skimage.morphology import square
import skimage.morphology

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

    def convert_pix_to_real(self, pixel_point):
        v, u = pixel_point
        u1 = u*self.res
        v1 = v*self.res
        x = u1 + self.pos.x
        y = v1 + self.pos.y
        world_point = (x, y)
        return world_point

    def convert_real_to_pix(self, world_point):
        x, y = world_point
        x1 = x - self.pos.x
        y1 = y - self.pos.y
        u = x1 / self.res
        v = y1 / self.res
        return (int(np.round(u)), int(np.round(v)))
    
    def change_val(self, val):
        if val > 50:
            return 1


    def map_cb(self, msg):
        data = msg.data
        self.res = msg.info.resolution
        self.ori = msg.info.origin.orientation
        self.pos = msg.info.origin.position
        self.wid =  msg.info.width
        self.hei =  msg.info.height

        
        #print("d", data)
        #print("d len1", type(data), len(data))
        
        print("r", self.res)
        print("o", self.ori)
        print("p", self.pos) 
        
        #print("data whole", data)

        

        arr = np.array(data, dtype = np.uint8)
        arr = np.reshape(arr, (self.hei, self.wid))
        #print("hei", self.hei, "wid", self.wid)

        uni, counts = np.unique(arr, return_counts=True)
        #print("counts", dict(zip(uni, counts)))

        #change_val_vec = np.vectorize(self.change_val)
        #arr_ones = self.change_val(arr)


        #arr[(arr < 50)] = 0
        #arr[(arr >= 50)] = 1

        #print("arr whole", arr)
        #print("arr part", arr[1::2, 1::2])
        

        arr_dil = skimage.morphology.dilation(arr, square(3))

        #print("dil whole", arr_dil)
        #print("dil part", arr[50])

        bright_pixel = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 100, 0, 0, 59, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 100, 0, 0],
                         [0, 49, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]
                         ], dtype=np.uint8)
        
    
        new_pixel = skimage.morphology.dilation(bright_pixel, square(3))

        #print("dilated",new_pixel)


        self.map = arr
        print(self.map.shape)
        


    def odom_cb(self, msg):
        pos = msg.pose.pose.position
        x = pos.x
        y = pos.y
        self.cur_pos = (x, y)        


    def goal_cb(self, msg):
        
        pos = msg.pose.position
        x = pos.x
        y = pos.y
        print("goal cb called")
        self.plan_path(self.cur_pos, (x, y), self.map)

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##
        print("real world start point", start_point)
        pixel_start_point = self.convert_real_to_pix(start_point)
        pixel_end_point = self.convert_real_to_pix(end_point)

        pixel_start_point = (pixel_start_point[1], pixel_start_point[0])
        pixel_end_point = (pixel_end_point[1], pixel_end_point[0])
        trajectory = self.a_star(pixel_start_point, pixel_end_point, map)
        self.trajectory.points = trajectory

        print("traj", trajectory)
        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz(duration=5.0)
    
    def check_valid(self, neighbors, map):
        valid_neighbors = []
        for neighbor in neighbors:
            if 0 <= neighbor[0] < map.shape[0] and 0 <= neighbor[1] < map.shape[1] and map[neighbor] != 100:
                valid_neighbors.append(neighbor)
        return valid_neighbors

    def backtrack(self, cameFrom, end_point):
        trajectory = []
        cur_point = end_point
        while cur_point is not None:
            trajectory.append(cur_point)
            cur_point = cameFrom[cur_point]
        trajectory.reverse()
        world_trajectory = [self.convert_pix_to_real(trajectory[i]) for i in range(len(trajectory))]
        print("FIRST TWO POINTS OF TRAJECTORY", world_trajectory[:2])
        return world_trajectory

    def a_star(self, start_point, end_point, map):
        def h(point):
            return np.linalg.norm(np.asarray(point) - np.asarray(end_point))
        
        gscore = np.ones(map.shape) * np.inf
        cameFrom = {start_point : None}
        openSet = heapdict()
        gscore[start_point] = 0
        
        openSet[start_point] = h(start_point)

        while len(openSet) > 0:
            current, _ = openSet.popitem()
            if current == end_point:
                return self.backtrack(cameFrom, end_point)
            
            neighbors = [(current[0] + 1, current[1]), (current[0] - 1, current[1]), (current[0], current[1] + 1), (current[0], current[1] - 1)]
            neighbors = self.check_valid(neighbors, map)
            for neighbor in neighbors:
                tentative_gscore = gscore[current] + 1
                if tentative_gscore < gscore[neighbor]:
                    gscore[neighbor] = tentative_gscore
                    fscore = tentative_gscore + h(neighbor)
                    cameFrom[neighbor] = current
                    if neighbor not in openSet:
                        openSet[neighbor] = fscore
                

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
