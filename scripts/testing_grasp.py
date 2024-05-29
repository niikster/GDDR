#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from testing_kuka.msg import GraspDetectionResult
import open3d as o3d
import pyransac3d as pyrsc
import numpy as np
import time
import struct
from itertools import combinations

class GraspDetectionNode:
    def __init__(self):
        rospy.init_node('grasp_detection_node')
        # self.input_topic = '/y/point_cloud_xyzrgb/points'
        self.input_topic = '/camera/depth_registered/points'

        self.pointcloud_sub = rospy.Subscriber(self.input_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
        self.plane_pub = rospy.Publisher('/y/planes', PointCloud2, queue_size=1)
        self.normals_pub = rospy.Publisher('/y/normals', MarkerArray, queue_size=1)
        self.contours_pub = rospy.Publisher('/y/contours', PointCloud2, queue_size=1)
        self.grasp_detection_pub = rospy.Publisher('/y/grasp_detection_result', GraspDetectionResult, queue_size=1)
        self.grasp_hypotheses_pub = rospy.Publisher('/y/grasp_hypotheses', MarkerArray, queue_size=1)

        self.input_frame_id = None
        self.marker_id = 0  # Initialize marker_id
        self.colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
            [0.0, 0.5, 0.5],  # Teal
            [0.5, 0.5, 0.0],  # Olive
        ]
        self.color_index = 0
        
    def pointcloud_callback(self, msg):
        self.input_frame_id = msg.header.frame_id

        start_time = time.time()

        cloud = self.ros_to_open3d(msg)
        if len(cloud.points) == 0:
            rospy.logwarn("Received an empty point cloud. Skipping...")
            return

        # Detect all planes, normals, contours on a point cloud
        planes, normals, contours, planes_array, contours_array, grasp_hypotheses_marker_array = self.detect_grasp_areas(cloud)

        # Clear previous markers
        self.clear_markers()

        # 
        grasp_detection_result_msg = self.create_grasp_detection_result_msg(planes_array, contours_array, normals)

        self.grasp_detection_pub.publish(grasp_detection_result_msg)
        self.plane_pub.publish(self.open3d_to_ros(planes))
        self.contours_pub.publish(self.open3d_to_ros(contours))
        self.normals_pub.publish(normals)
        self.grasp_hypotheses_pub.publish(grasp_hypotheses_marker_array)

        end_time = time.time()
        rospy.loginfo(f"Inference took {end_time - start_time:.2f} seconds")

    def clear_markers(self):
        clear_marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        clear_marker_array.markers.append(clear_marker)
        self.normals_pub.publish(clear_marker_array)
        self.grasp_hypotheses_pub.publish(clear_marker_array)

    def ros_to_open3d(self, ros_cloud):
        points_list = list(pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True))
        if not points_list:
            return o3d.geometry.PointCloud()
        
        points = np.array(points_list)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        return cloud

    def open3d_to_ros(self, cloud):
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else None
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.input_frame_id

        if colors is not None:
            fields = [
                pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1)
            ]
            cloud_data = []
            for i in range(points.shape[0]):
                x, y, z = points[i]
                r, g, b = colors[i]
                rgb = struct.unpack('I', struct.pack('BBBB', int(b * 255), int(g * 255), int(r * 255), 255))[0]
                cloud_data.append([x, y, z, rgb])
            ros_msg = pc2.create_cloud(header, fields, cloud_data)
        else:
            ros_msg = pc2.create_cloud_xyz32(header, points)
        return ros_msg

    def create_grasp_detection_result_msg(self, planes_array, contours_array, marker_array):
        result_msg = GraspDetectionResult()
        result_msg.header.stamp = rospy.Time.now()
        result_msg.header.frame_id = self.input_frame_id

        for pc in planes_array:
            result_msg.planes.append(self.open3d_to_ros(pc))
        for contour in contours_array:
            result_msg.contours.append(self.open3d_to_ros(contour))
        result_msg.markers = marker_array

        return result_msg
    

    def find_grasp_hypotheses(self, hull_points, plane_normal, color, grasp_marker_array):
        grasp_hypotheses = []
        boundary_points_np = np.asarray(hull_points.points)

        # Use RANSAC to fit lines to the points
        lines = []
        while len(boundary_points_np) > 2:
            ln = pyrsc.Line()
            a, b, inliers = ln.fit(boundary_points_np, thresh=0.01, maxIteration=1000)
            if len(inliers) < 2:
                break
            line_points = boundary_points_np[inliers]
            lines.append(line_points)
            boundary_points_np = np.delete(boundary_points_np, inliers, axis=0)

        # Find pairs of parallel lines
        for line1, line2 in combinations(lines, 2):
            vector1 = line1[1] - line1[0]
            vector2 = line2[1] - line2[0]
            vector1_normalized = vector1 / np.linalg.norm(vector1)
            vector2_normalized = vector2 / np.linalg.norm(vector2)
            dot_product = np.dot(vector1_normalized, vector2_normalized)

            if abs(dot_product) > 0.9:  # Nearly parallel lines
                for p1 in line1:
                    for p2 in line2:
                        distance = np.linalg.norm(p1 - p2)
                        if distance <= 0.07:  # 7 cm
                            midpoint = (p1 + p2) / 2
                            grasp_hypotheses.append((p1, p2, midpoint, plane_normal))

                            marker = Marker()
                            marker.header.frame_id = self.input_frame_id
                            marker.header.stamp = rospy.Time.now()
                            marker.ns = "grasps"
                            marker.id = self.marker_id
                            self.marker_id += 1
                            marker.type = Marker.ARROW
                            marker.action = Marker.ADD
                            marker.pose.orientation.w = 1.0
                            marker.points = [Point(*midpoint), Point(*(midpoint + plane_normal * 0.1))]
                            marker.scale.x = 0.01
                            marker.scale.y = 0.02
                            marker.scale.z = 0.01
                            marker.color.a = 1.0
                            marker.color.r = color[0]
                            marker.color.g = color[1]
                            marker.color.b = color[2]
                            grasp_marker_array.markers.append(marker)

        return lines, grasp_hypotheses


    def process_plane(self, plane_cloud, color):
        plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))

        average_normal = np.mean(np.asarray(plane_cloud.normals), axis=0)
        average_normal = average_normal / np.linalg.norm(average_normal)

        points = np.asarray(plane_cloud.points)
        jittered_points = points + np.random.normal(0, 1e-6, points.shape)
        jittered_cloud = o3d.geometry.PointCloud()
        jittered_cloud.points = o3d.utility.Vector3dVector(jittered_points)
        
        try:
            hull, _ = jittered_cloud.compute_convex_hull()
            hull.paint_uniform_color(color)  # Apply uniform color to the hull
            hull_points = o3d.geometry.PointCloud()
            hull_points.points = hull.vertices
            hull_points.colors = o3d.utility.Vector3dVector(np.tile(color, (len(hull.vertices), 1)))  # Apply color to points
            boundary_points = hull.sample_points_poisson_disk(len(plane_cloud.points) // 10)
            boundary_points.colors = o3d.utility.Vector3dVector(np.tile(color, (len(hull_points.points), 1)))
        except Exception as e:
            boundary_points = None
            rospy.logwarn(f"Convex hull computation failed: {e}")


        marker = Marker()
        marker.header.frame_id = self.input_frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "normals"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        centroid = np.mean(np.asarray(plane_cloud.points), axis=0)
        start_point = Point(*centroid)
        end_point = Point(*(centroid + average_normal * 0.1))
        marker.points = [start_point, end_point]

        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]


        return plane_cloud, marker, boundary_points, average_normal

    def detect_grasp_areas(self, cloud):
        cloud = cloud.voxel_down_sample(voxel_size=0.01)

        planes = o3d.geometry.PointCloud()
        contours = o3d.geometry.PointCloud()
        plane_normal_array = MarkerArray()
        grasp_marker_array = MarkerArray()
        planes_array = []
        contours_array = []
        grasp_candidates = []


        while True:
            plane_model, inliers = cloud.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=2000)
            if len(inliers) < 20:
                break
            plane_cloud = cloud.select_by_index(inliers)


            # plane_normal = plane_model[:3]
            # if abs(plane_normal[2]) > 0.80:  # Assuming the floor is roughly horizontal
            # if len(inliers) > 3000:
            #     cloud = cloud.select_by_index(inliers, invert=True)
            #     continue

            # Use Open3D's DBSCAN clustering to separate disconnected components
            labels = np.array(plane_cloud.cluster_dbscan(eps=0.02, min_points=10))
            max_label = labels.max()

            for i in range(max_label + 1):
                cluster_indices = np.where(labels == i)[0]
                cluster = plane_cloud.select_by_index(cluster_indices)
                
                # Use predefined colors
                color = self.colors[self.color_index % len(self.colors)]
                self.color_index += 1

                plane_colors = np.tile(color, (len(cluster.points), 1))
                cluster.colors = o3d.utility.Vector3dVector(plane_colors)

                cluster, marker, boundary_points, average_normal = self.process_plane(cluster, color)

                planes += cluster

                planes += cluster
                if boundary_points is not None:
                    boundary_points.colors = o3d.utility.Vector3dVector(np.tile(color, (len(boundary_points.points), 1)))
                    contours += boundary_points
                    contours_array.append(boundary_points)
                    edges, grasp_candidates = self.find_grasp_hypotheses(boundary_points, average_normal, color, grasp_marker_array)

                plane_normal_array.markers.append(marker)
                planes_array.append(cluster)
                    
            cloud = cloud.select_by_index(inliers, invert=True)

        return planes, plane_normal_array, contours, planes_array, contours_array, grasp_marker_array
                    

if __name__ == '__main__':
    try:
        node = GraspDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass