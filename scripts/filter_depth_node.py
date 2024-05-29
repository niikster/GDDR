#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from testing_kuka.msg import YoloResult

class DepthFilter:
    def __init__(self):
        # BE WARY OF MSG ENCODING AND FRAME ID
        # /camera/depth/image_rect -- 32FC1
        # /camera/depth/image_raw -- 16UC1
        # to check topic info type: 
        # rostopic echo -n 1 --noarr  /camera/depth/image_rect
        self.input_depth_topic = rospy.get_param("~input_depth_topic", '/camera/depth/image_rect')
        self.input_masks_topic = rospy.get_param("~input_masks_topic", '/yolo_result')
        self.output_topic = rospy.get_param("~output_topic", "/y/filtered_depth")
        
        # Initialize
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber(self.input_depth_topic, Image, self.depth_cb, queue_size=1)
        self.mask_sub = rospy.Subscriber(self.input_masks_topic, YoloResult, self.mask_cb, queue_size=1)
        self.filtered_depth_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.depth_img = None
        self.header = None # create Header to copy input_depth Header info

    def depth_cb(self, depth_msg:Image):
        # Create header to specify frame_id
        self.header = depth_msg.header
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

    def mask_cb(self, mask_msg):
        if self.depth_img is None:
            return
        masks = mask_msg.masks

        # Create a binary mask image
        mask_img = np.zeros((self.depth_img.shape[0], self.depth_img.shape[1]), dtype=np.uint8)
        for mask in masks:
            mask = self.bridge.imgmsg_to_cv2(mask, desired_encoding="mono8")
            mask_img[mask == 255] = 255

        # Filter out depth data based on the segmentation mask
        filtered_depth_img = self.depth_img.copy()
        filtered_depth_img[mask_img == 0] = 0

        # Update time stamp of header for publishing
        self.header.stamp = rospy.Time.now()

        # Create resulting imgmsg
        # encoding==passthrough means the same encoding as 
        res_msg = self.bridge.cv2_to_imgmsg(filtered_depth_img, encoding="passthrough")
        res_msg.header = self.header
        self.filtered_depth_pub.publish(res_msg)


if __name__ == '__main__':
    rospy.init_node('depth_filter_node')
    rospy.loginfo('Depth filter node has been started.')
    node = DepthFilter()
    rospy.spin()
    rospy.loginfo('Depth filter node has been killed.')