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
        self.parking_distance = 0
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
        poses for the trajectory. Returns intersection point of the closest trajectory segment
        after the `segment` index that intersects with a circle of radius `self.lookahead_dist`
        centered on pose.
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
            soln_param = 0
            if (0 <= t1 <= 1 and 0 <= t2 <= 1):
                # edge case, currently return t1
                soln_param = t1
                print("edge case!")
            elif (0 <= t1 <= 1):
                soln_param = t1
            elif (0 <= t2 <= 1):
                # t2 is our intersection point
                soln_param = t2
        if (soln_param == 0):
            # failure, no line segment within lookahead distance
            return np.array([-1, -1])
        return seg_start + soln_param * seg_v
    
    def lookahead_to_drive(self, pose, segment):
        """ Takes an odometry message of car location and index of nearest trajectory
        segment and publishes a drive command to navigate the car to the closest 
        line segment.
        """
        world_point = self.lookahead_intersection(pose, segment)
        displacement = world_point - pose
        # Convert to world frame
        quat = pose.pose.pose.orientation
        thetas = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        theta = thetas[2]
        rot_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        relative_displacement = np.matmul(np.linalg.inv(rot_matrix), displacement)
        self.pursuit_drive_callback(relative_displacement)

    
    def validPoint(self, point): 
        """
        Currently a failure detector for pursuit driving. Checks if
        point output by lookahead_intersection is invalid.
        """
        return not (point[0] == -1.0 and point[1] == -1.0)
    
    def pursuit_drive_callback(self, goal_point):
        """ Given a goal point, issues a drive command to the racecar to bring it closer.
        Goal point must be the relative distance from the front of the car.
        """
        self.relative_x = goal_point[0]
        self.relative_y = goal_point[1]
        drive_cmd = AckermannDriveStamped()
        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.header.frame_id = 'constant'
        drive_cmd.drive.speed = self.speed

        if self.reverse < self.reverse_time: # Check if reversing
            drive_cmd.drive.speed = self.reverse_speed
            self.reverse = self.reverse + 1
        else:
            theta = np.arctan2(self.relative_y, self.relative_x)
            dist = np.sqrt(self.relative_x**2+self.relative_y**2)

            if self.validPoint(goal_point):
                if abs(dist-self.parking_distance) < 0.05 and abs(theta) <= 0.05: # If within distance and angle tolerenace, park
                    drive_cmd.drive.speed = 0
                    drive_cmd.drive.acceleration = 0
                    drive_cmd.drive.jerk = 0
                elif dist > self.parking_distance: # If away from cone, control theta proportionally to align upon closing gap
                    if dist > self.parking_distance:
                        if theta != 0:
                            drive_cmd.drive.steering_angle = theta
                        else:
                            drive_cmd.drive.steering_angle = 0
                elif dist < self.parking_distance and abs(theta) <= 0.05: # If too close too cone but aligned, back up
                    drive_cmd.drive.steering_angle = 0
                    drive_cmd.drive.speed = self.reverse_speed
                else: # Too close and not aligned, reverse for a few timesteps then retry
                    drive_cmd.drive.speed = self.reverse_speed
                    self.reverse = 0           
            else: # Cone not within FOV, drive in a circle to find
                drive_cmd.drive.steering_angle = 0.34 # max steering angle
                # This can get stuck in a circle if the cone is placed directly to the left of the wheel
                # as a result of max turning radius and camera FOV. This may be resolved if the actual 
                # camera has a wider FOV, though this could be solved in code by iterating a value everytime
                # this runs and setting it to 0 otherwise. If the value is exceeded, circle times out and a right
                # steer should be briefly published to offset circle and find the cone. I'm too lazy to implement
                # it rn though lol.
        









        


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
