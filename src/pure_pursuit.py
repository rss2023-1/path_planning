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
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from std_msgs.msg import Float32

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic", "/pf/pose/odom")
        #self.odom_topic       = "/odom"
        self.lookahead        = rospy.get_param("~lookahead_dist", 1.0)
        self.speed            = rospy.get_param("~pursuit_speed", 0.5)
        self.wheelbase_length = 0.2 #measure this for sure
        self.parking_distance = 0
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.viz_point = rospy.Publisher("/vizpoint", Marker, queue_size=1)
        self.error_pub = rospy.Publisher("/error", Float32, queue_size=1)
        self.coordinates = []
        self.reverse_time = 5
        self.reverse = self.reverse_time 
        self.parking_distance = .1 # meters; try playing with this number!
        self.reverse_speed = -0.5


    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        rospy.loginfo("Receiving new trajectory:" + str(len(msg.poses)) + "points")
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.coordinates = []
        # Read trajectory x and y values when initialized
        # might want to extract pose directions as well? or is just x,y sufficient
        for pose in msg.poses:
            x = pose.position.x
            y = pose.position.y
            self.coordinates.append(np.asarray([x, y]))

        self.coordinates = np.stack(self.coordinates)
        self.endpoint = self.coordinates[-1]


    def odometry_callback(self, odom):
        # Initialize vector containing all line segment points
        line_segments = np.zeros((self.coordinates.shape[0] - 1, 4))
        line_segments[:, :2] = self.coordinates[:-1, :]
        line_segments[:, 2:4] = self.coordinates[1:, :]

        #line_segments is a 4 x coordinates - 1 array, structured as follows:
        # [x0, y0, x1, y1] where (x0, y0) is the coordinate of the first point in the line segment and (x1, y1) is the second point

        # Get baselink position as x, y coord
        xpos = odom.pose.pose.position.x
        ypos = odom.pose.pose.position.y
        point = np.array([xpos, ypos])
        
        # Calculate minimum distances between current position and each line segment
        # Initialize array
        min_distances = np.zeros(line_segments.shape[0])
        min_points = np.zeros((line_segments.shape[0], 2))
        # Loop over each line segment
        for i in range(line_segments.shape[0]):
            # Get the starting and ending points of the line segment
            p1 = line_segments[i, :2]
            p2 = line_segments[i, 2:]
            # Compute the vector representing the line segment
            v = p2 - p1
            # Compute the vector from the starting point of the line segment to the point
            w = point - p1
            # Compute the projection of w onto v
            proj = np.dot(w, v) / np.dot(v, v) * v
            min_points[i] = proj
            # Compute the distance between the point and the projected point
            dist = np.linalg.norm(proj - w)
            # If the projection is outside the line segment, compute the distance to the closest endpoint
            if np.dot(proj - p1, proj - p2) > 0:
                dist = min(np.linalg.norm(point - p1), np.linalg.norm(point - p2))
                if np.linalg.norm(point - p1) < np.linalg.norm(point - p2):
                    min_points[i] = p1
                else:
                    min_points[i] = p2
            # Store the minimum distance for this line segment
            min_distances[i] = dist
        
        indices = np.argsort(min_distances)
        # Publish for error metric
        min_dist = min(min_distances)
        self.error_pub.publish(min_dist)
        goal_point = self.lookahead_intersection(point, indices, line_segments, odom.pose.pose)
        if self.validPoint(goal_point):
            robot_point = self.world_to_robot(odom.pose.pose, goal_point)
            print("GOAL POINT", goal_point, "ROBOT POINT", robot_point)
        else:
            robot_point = goal_point
        self.pursuit_drive_callback(robot_point)
        
    def distance_to_goal(self, point, goal):
        return np.sqrt((point[1] - goal[1])**2 + (point[0] - goal[0])**2)

    def lookahead_intersection(self, point, indices, line_segments, robot_pose):
        """
        Takes in pose as Odometry msg and a line segment index in the list of 
        poses for the trajectory. Returns intersection point of the closest trajectory segment
        after the `segment` index that intersects with a circle of radius `self.lookahead_dist`
        centered on pose.
        Intersection equation: 
        https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428
        """
        Q = point
        r = self.lookahead
        current = 0
        # We iterate through subsequent line segments after closest until intersection
        sorted_line_segments = line_segments[indices]
        while current < sorted_line_segments.shape[0]:
            cur_index = indices[current]

            P1 = sorted_line_segments[current, :2]     # Start of line segment
            V = sorted_line_segments[current, 2:]   - P1  # Vector along line segment

            a = np.dot(V, V)
            b = 2 * np.dot(V, P1 - Q)
            c = P1.dot(P1) + Q.dot(Q) - 2 * P1.dot(Q) - r**2
            disc = b**2 - 4 * a * c

            if disc > 0:
                sqrt_disc = np.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)
                distance_t1 = np.inf
                distance_t2 = np.inf
                point1 = None
                point2 = None
                if 0 <= t2 <= 1:
                    point2 = P1 + t2 * V
                    relative_pt2 = self.world_to_robot(robot_pose, point2)
                    if (relative_pt2[0] > 0):
                        distance_t2 = self.distance_to_goal(point2, line_segments[-1, 2:4])

                if 0 <= t1 <= 1:
                    point1 = P1 + t1 * V
                    relative_pt1 = self.world_to_robot(robot_pose, point1)
                    if (relative_pt1[0] > 0):
                        distance_t1 = self.distance_to_goal(point1, line_segments[-1, 2:4])
                #print("heuristic distance", line_segments[cur_index + 1, :2])
                # Heuristic: parameter that describes a point "farther along" the trajectory
                distances = [distance_t1, distance_t2]
                points = [point1, point2]
                if distances[np.argmin(distances)] < np.inf:
                    print("bruh")
                    self.publish_point(points[np.argmin(distances)])
                    return points[np.argmin(distances)]

            current += 1
        rospy.loginfo("FAILED")
        r += 0.5
        return [-1, -1]

    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = rospy.Time.now()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

    def publish_point(self, point):
        marker = Marker()
        marker.header = self.make_header("/map")
        marker.ns = "/vizpoint"
        marker.id = 0
        marker.type = 2 # sphere
        #marker.lifetime = rospy.Duration.from_sec(duration)
        marker.action = 0
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.viz_point.publish(marker)
    
    def world_to_robot(self, robot_pose, world_point):
        """ Takes an odometry message of car location and goal point on closest  and publishes a drive command to navigate the car to the closest 
        line segment.
        """
        displacement = np.asarray([world_point[0] - robot_pose.position.x, world_point[1] - robot_pose.position.y])
        # Convert to world frame
        rot_matrix = self.rot_matrix(robot_pose)
        robot_point = np.matmul(rot_matrix, displacement)
        return robot_point

    def rot_matrix(self, robot_pose):
        quat = robot_pose.orientation
        thetas = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        theta = thetas[2]
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

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

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd

        if self.reverse < self.reverse_time: # Check if reversing
            drive_cmd.drive.speed = self.reverse_speed
            self.reverse = self.reverse + 1
        else:
            theta = np.arctan2(self.relative_y, self.relative_x)
            dist = np.sqrt(self.relative_x**2+self.relative_y**2)

            if self.validPoint(goal_point):
                print("THETA", theta)
                if abs(dist-self.parking_distance) < 0.05 and abs(theta) <= 0.05: # If within distance and angle tolerenace, park
                    drive_cmd.drive.speed = 0
                    drive_cmd.drive.acceleration = 0
                    drive_cmd.drive.jerk = 0
                elif dist > self.parking_distance: # If away from cone, control theta proportionally to align upon closing gap
                    drive_cmd.drive.steering_angle = theta
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
        self.drive_pub.publish(drive_cmd)









        


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
